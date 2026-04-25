"""
MedSentinel API Server
=======================

FastAPI server that exposes the MedSentinel backend to the React UI.

Endpoints:
  POST /diagnose   — Run a full diagnosis episode (UI calls this)
  GET  /health     — Health check
  GET  /patients   — Get sample patient cases from the dataset

This wraps the existing OpenEnv server app and adds the /diagnose endpoint
that the React UI needs. The OpenEnv endpoints (/reset, /step, /state) are
still available alongside.

Run:
  uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload

Or with the start script:
  python api_server.py
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from typing import Any, Dict, List, Optional

# Repo root on path
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agents.auditor_agent import audit_doctor_output
from agents.clinical_verification_layer import ClinicalVerificationLayer
from agents.doctor_agent import DoctorAgent
from env.medsentinel_env import EnvConfig, MedSentinelEnv
from env.reward_system import compute_reward
from env.schema_drift import apply_schema_drift
from tools.mcp_tools import (
    check_allergies,
    dose_check,
    drug_interactions,
    icd_lookup,
    query_labs,
)

# ─── App setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="MedSentinel API",
    description="Multi-agent medical RL backend for MedSentinel UI",
    version="3.0.0",
)

# Allow all origins for local dev and HuggingFace Spaces
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared env config
_DATASET_PATH = os.path.join(_REPO_ROOT, "data", "patient_cases.json")
_DRUG_DB_PATH  = os.path.join(_REPO_ROOT, "data", "emergency_drugs.json")
_ICD_DB_PATH   = os.path.join(_REPO_ROOT, "data", "icd10_emergency_conditions.json")

# ─── Request / Response models ────────────────────────────────────────────────

class VitalsInput(BaseModel):
    bp_systolic: float = 120
    bp_diastolic: float = 80
    heart_rate: float = 75
    temperature: float = 37.0
    spo2: float = 98
    respiratory_rate: float = 16

class LabsInput(BaseModel):
    troponin_i: float = 0.0
    bnp: float = 0.0
    creatinine: float = 1.0
    glucose: float = 100.0
    wbc: float = 7.0
    hemoglobin: float = 14.0

class DiagnoseRequest(BaseModel):
    patientId: str = Field(default="P-001")
    age: int = Field(default=50, ge=1, le=120)
    gender: str = Field(default="Male")
    chiefComplaint: str = Field(default="")
    vitals: VitalsInput = Field(default_factory=VitalsInput)
    labs: LabsInput = Field(default_factory=LabsInput)
    allergies: List[str] = Field(default_factory=list)
    medications: List[str] = Field(default_factory=list)
    safeDrugs: Optional[List[str]] = None
    unsafeDrugs: Optional[List[str]] = None
    groundTruthDiagnosis: Optional[str] = None
    driftEnabled: bool = True
    driftProbability: float = Field(default=35.0, ge=0, le=100)
    seed: int = 42


class DriftRename(BaseModel):
    section: str
    original: str
    renamed: str


class ToolCall(BaseModel):
    name: str
    input: Dict[str, Any]
    output: Dict[str, Any]
    verdict: str


class DoctorOutput(BaseModel):
    icd10: str
    diagnosisName: str
    drug: str
    dose: str
    confidence: float
    schemaDriftHandled: bool
    reasoning: str


class AuditorOutput(BaseModel):
    safe: bool
    flags: List[str]
    notes: List[str]


class RewardComponent(BaseModel):
    label: str
    value: float


class RewardOutput(BaseModel):
    total: float
    components: List[RewardComponent]


class CVLOutput(BaseModel):
    verified: bool
    changes: List[str]
    riskFlags: List[str]
    notes: str
    fallback: bool


class DiagnoseResponse(BaseModel):
    drift: Dict[str, Any]
    doctor: DoctorOutput
    toolCalls: List[ToolCall]
    auditor: AuditorOutput
    reward: RewardOutput
    cvl: Optional[CVLOutput] = None


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _build_patient_dict(req: DiagnoseRequest) -> Dict[str, Any]:
    """Convert DiagnoseRequest to the patient dict format the backend expects."""
    patient: Dict[str, Any] = {
        "patient_id": req.patientId or "P-001",
        "age": req.age,
        "gender": req.gender,
        "chief_complaint": req.chiefComplaint,
        "vitals": {
            "bp_systolic": req.vitals.bp_systolic,
            "bp_diastolic": req.vitals.bp_diastolic,
            "heart_rate": req.vitals.heart_rate,
            "temperature": req.vitals.temperature,
            "spo2": req.vitals.spo2,
            "respiratory_rate": req.vitals.respiratory_rate,
        },
        "lab_results": {
            "troponin_i": req.labs.troponin_i,
            "bnp": req.labs.bnp,
            "creatinine": req.labs.creatinine,
            "glucose": req.labs.glucose,
            "wbc": req.labs.wbc,
            "hemoglobin": req.labs.hemoglobin,
        },
        "known_allergies": req.allergies,
        "current_medications": req.medications,
    }

    if req.safeDrugs is not None:
        patient["safe_drugs"] = req.safeDrugs
    else:
        # Build safe_drugs from drug DB based on diagnosis category
        patient["safe_drugs"] = []

    if req.unsafeDrugs is not None:
        patient["unsafe_drugs"] = req.unsafeDrugs
    else:
        # Build unsafe_drugs = known allergies
        patient["unsafe_drugs"] = list(req.allergies)

    if req.groundTruthDiagnosis:
        patient["ground_truth_diagnosis"] = req.groundTruthDiagnosis

    return patient


def _run_mcp_tools(
    patient: Dict[str, Any],
    drug: str,
    dose: Optional[float],
    drift_occurred: bool,
    drift_changes: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Run all 5 MCP tools and return their logs in UI format."""
    tool_logs = []

    # Schema normalizer (if drift occurred)
    if drift_occurred:
        renames = []
        for old_k, new_k in drift_changes.get("vitals", {}).items():
            renames.append(f"{old_k}→{new_k}")
        for old_k, new_k in drift_changes.get("lab_results", {}).items():
            renames.append(f"{old_k}→{new_k}")
        tool_logs.append({
            "name": "schema_normalizer",
            "input": {"renamed_keys": renames},
            "output": {"resolved": True, "mappings": len(renames)},
            "verdict": "drift",
        })

    # query_labs
    labs_result = query_labs(patient)
    tool_logs.append({
        "name": "query_labs",
        "input": {"patient_id": patient.get("patient_id")},
        "output": labs_result,
        "verdict": "drift" if labs_result.get("drift_detected") else "safe",
    })

    # check_allergies
    if drug:
        allergy_result = check_allergies(patient, drug)
        tool_logs.append({
            "name": "check_allergies",
            "input": {"drug_name": drug, "patient_allergies": patient.get("known_allergies", [])},
            "output": allergy_result,
            "verdict": "unsafe" if allergy_result.get("verdict") == "unsafe" else "safe",
        })

    # dose_check
    if drug and dose is not None:
        dose_result = dose_check(drug, dose)
        tool_logs.append({
            "name": "dose_check",
            "input": {"drug_name": drug, "dose_mg": dose},
            "output": dose_result,
            "verdict": "unsafe" if not dose_result.get("in_range", True) else "safe",
        })

    # drug_interactions
    meds = patient.get("current_medications", [])
    if drug and meds:
        interaction_result = drug_interactions(drug, meds)
        tool_logs.append({
            "name": "drug_interactions",
            "input": {"drug": drug, "meds": meds},
            "output": interaction_result,
            "verdict": "warning" if interaction_result.get("has_conflict") else "safe",
        })

    # icd_lookup
    gt_dx = patient.get("ground_truth_diagnosis", "")
    if gt_dx:
        icd_result = icd_lookup(gt_dx)
        tool_logs.append({
            "name": "icd_lookup",
            "input": {"code": gt_dx},
            "output": icd_result,
            "verdict": "safe",
        })

    return tool_logs


def _dose_display(drug: str, dose_mg: Optional[float]) -> str:
    """Format dose as a display string matching what the UI expects."""
    if dose_mg is None:
        return "—"
    units: Dict[str, str] = {
        "nitroglycerin": "mg sublingual",
        "aspirin": "mg PO",
        "heparin": "units IV",
        "morphine": "mg IV",
        "metoprolol": "mg PO",
        "insulin": "units IV",
        "ceftriaxone": "mg IV",
        "vancomycin": "mg IV",
        "piperacillin-tazobactam": "mg IV",
        "epinephrine": "mg IM",
        "naloxone": "mg IV",
        "diazepam": "mg IV",
        "furosemide": "mg IV",
        "amiodarone": "mg IV",
        "alteplase": "mg IV",
        "labetalol": "mg IV",
        "magnesium-sulfate": "mg IV",
    }
    unit = units.get(drug.lower(), "mg")
    return f"{dose_mg} {unit}"


def _build_reward_components(breakdown: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert backend reward breakdown to UI format."""
    components = []
    label_map = {
        "diagnosis":   "Correct ICD-10 diagnosis",
        "safe_drug":   "Safe drug prescribed",
        "dosage":      "Correct dosage",
        "drift":       "Schema drift handled",
        "auditor":     "Auditor approved",
    }
    penalty_map = {
        "allergy":    "Allergic drug penalty",
        "wrong_dx":   "Wrong diagnosis (confident)",
    }

    comp_dict = breakdown.get("components", {})
    for k, v in comp_dict.items():
        label = label_map.get(k, k.replace("_", " ").title())
        if v != 0:
            components.append({"label": label, "value": float(v)})

    pen_dict = breakdown.get("penalties", {})
    for k, v in pen_dict.items():
        label = penalty_map.get(k, k.replace("_", " ").title())
        if v != 0:
            components.append({"label": label, "value": float(v)})

    return components


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Health check — UI polls this to check if backend is running."""
    return {
        "status": "ok",
        "version": "3.0.0",
        "dataset": os.path.exists(_DATASET_PATH),
        "drug_db": os.path.exists(_DRUG_DB_PATH),
        "icd_db": os.path.exists(_ICD_DB_PATH),
    }


@app.get("/patients")
def get_patients(n: int = 10, mode: str = "test"):
    """Return n sample patient cases from the dataset for the UI to display."""
    try:
        with open(_DATASET_PATH) as f:
            cases = json.load(f)
        # Return basic info only (no ground truth diagnosis)
        result = []
        for c in cases[:n]:
            result.append({
                "patient_id": c.get("patient_id"),
                "age": c.get("age"),
                "gender": c.get("gender"),
                "chief_complaint": c.get("chief_complaint"),
            })
        return {"patients": result, "total": len(cases)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/diagnose", response_model=DiagnoseResponse)
def diagnose(req: DiagnoseRequest):
    """
    Run a full MedSentinel diagnosis episode.

    This is the main endpoint the React UI calls instead of diagnosisEngine.ts mock.

    Pipeline:
      1. Build patient dict from request
      2. Apply schema drift (if enabled)
      3. Run doctor agent (local rule-based or Anthropic API)
      4. Run MCP tools
      5. Run auditor
      6. Compute reward
      7. Run CVL (if API key available)
      8. Return structured response matching DiagnoseResponse
    """
    try:
        # ── 1. Build patient dict ──────────────────────────────────────────
        patient_original = _build_patient_dict(req)

        # ── 2. Apply schema drift ─────────────────────────────────────────
        drift_occurred = False
        drift_changes: Dict[str, Any] = {"vitals": {}, "lab_results": {}}

        if req.driftEnabled:
            patient_observed, drift_occurred, drift_changes = apply_schema_drift(
                patient_original,
                seed=req.seed,
                drift_probability=req.driftProbability / 100.0,
                max_key_renames_per_section=2,
            )
        else:
            patient_observed = dict(patient_original)

        # ── 3. Doctor agent ───────────────────────────────────────────────
        # Use Anthropic if key is available, otherwise rule-based local
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if api_key:
            try:
                doctor = DoctorAgent(provider="anthropic", seed=req.seed)
            except Exception:
                doctor = DoctorAgent(provider="local", seed=req.seed)
        else:
            doctor = DoctorAgent(provider="local", seed=req.seed)

        doctor_output = doctor.diagnose(patient_observed)

        drug       = doctor_output.get("prescribed_drug", "")
        dose_mg    = doctor_output.get("dosage_mg")
        icd10      = doctor_output.get("diagnosis_icd10", "")
        dx_name    = doctor_output.get("diagnosis_name", "")
        confidence = float(doctor_output.get("confidence", 0.0))
        reasoning  = doctor_output.get("reasoning", "")
        drift_handled = bool(doctor_output.get("schema_drift_handled", False))

        # ── 4. MCP tools ──────────────────────────────────────────────────
        tool_logs = _run_mcp_tools(
            patient_observed, drug, dose_mg, drift_occurred, drift_changes
        )

        # ── 5. Auditor ────────────────────────────────────────────────────
        auditor = audit_doctor_output(
            doctor_output,
            patient_observed,
            drug_db_path=_DRUG_DB_PATH,
        )
        auditor_flags = {
            "is_correct": bool(auditor.get("safe", False)),
            "flags": list(auditor.get("flags", [])),
        }

        # ── 6. Reward ─────────────────────────────────────────────────────
        reward_float, breakdown = compute_reward(
            doctor_output,
            patient_observed,
            auditor_flags=auditor_flags,
            drift_flag=bool(drift_occurred),
            drug_db_path=_DRUG_DB_PATH,
            icd_db_path=_ICD_DB_PATH,
        )

        reward_components = _build_reward_components(breakdown)

        # ── 7. CVL ────────────────────────────────────────────────────────
        cvl_data: Optional[CVLOutput] = None
        if api_key:
            try:
                cvl = ClinicalVerificationLayer()
                if cvl.is_active:
                    cvl_result = cvl.verify(
                        patient_original=patient_original,
                        patient_observed=patient_observed,
                        doctor_output=doctor_output,
                        auditor_flags=auditor_flags,
                    )
                    cvl_data = CVLOutput(
                        verified=bool(cvl_result.get("cvl_verified", False)),
                        changes=list(cvl_result.get("cvl_changes", [])),
                        riskFlags=list(cvl_result.get("cvl_risk_flags", [])),
                        notes=str(cvl_result.get("cvl_notes", "")),
                        fallback=bool(cvl_result.get("cvl_fallback", False)),
                    )
            except Exception as cvl_err:
                cvl_data = CVLOutput(
                    verified=False, changes=[], riskFlags=[],
                    notes=f"CVL unavailable: {cvl_err}", fallback=True,
                )

        # ── 8. Build response ─────────────────────────────────────────────
        # Build drift renames list for UI
        drift_renames = []
        for orig_k, new_k in drift_changes.get("vitals", {}).items():
            drift_renames.append({"section": "vitals", "original": orig_k, "renamed": new_k})
        for orig_k, new_k in drift_changes.get("lab_results", {}).items():
            drift_renames.append({"section": "labs", "original": orig_k, "renamed": new_k})

        return DiagnoseResponse(
            drift={
                "occurred": bool(drift_occurred),
                "renames": drift_renames,
            },
            doctor=DoctorOutput(
                icd10=icd10,
                diagnosisName=dx_name,
                drug=drug,
                dose=_dose_display(drug, dose_mg),
                confidence=confidence,
                schemaDriftHandled=drift_handled,
                reasoning=reasoning,
            ),
            toolCalls=[
                ToolCall(
                    name=t["name"],
                    input=t["input"],
                    output=t["output"],
                    verdict=t["verdict"],
                )
                for t in tool_logs
            ],
            auditor=AuditorOutput(
                safe=bool(auditor.get("safe", False)),
                flags=list(auditor.get("flags", [])),
                notes=list(auditor.get("notes", [])),
            ),
            reward=RewardOutput(
                total=round(float(reward_float), 3),
                components=[
                    RewardComponent(label=c["label"], value=c["value"])
                    for c in reward_components
                ],
            ),
            cvl=cvl_data,
        )

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Diagnosis pipeline failed: {e}")


# ─── Dev runner ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting MedSentinel API on http://localhost:{port}")
    print(f"UI should be running on http://localhost:8080")
    print(f"ANTHROPIC_API_KEY: {'✅ set' if os.environ.get('ANTHROPIC_API_KEY') else '⚠️  not set (using local rule-based doctor)'}")
    uvicorn.run("api_server:app", host="0.0.0.0", port=port, reload=True)
