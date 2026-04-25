"""
MedSentinel MCP Tools
=====================
Model Context Protocol (MCP) tools are deterministic lookup functions
that a doctor agent can call during diagnosis to retrieve structured
clinical information.

Available tools:
- query_labs         : normalize lab results, detect schema drift
- check_allergies    : drug-allergy and unsafe list check
- drug_interactions  : interaction check against current meds
- icd_lookup         : ICD-10 code/name lookup
- dose_check         : dosage range validation

All tools are deterministic, read-only, and return typed dicts.
Use call_tool(name, **kwargs) for dispatch or import directly.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any, Dict, List, Mapping, Optional

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


@lru_cache(maxsize=1)
def _load_emergency_drugs() -> Dict[str, Dict[str, Any]]:
    path = os.path.join(_REPO_ROOT, "data", "emergency_drugs.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


@lru_cache(maxsize=1)
def _load_icd10_conditions() -> Dict[str, Dict[str, Any]]:
    path = os.path.join(_REPO_ROOT, "data", "icd10_emergency_conditions.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _as_float(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _normalized_list(values: Any) -> List[str]:
    if not isinstance(values, list):
        return []
    return [str(v).strip().lower() for v in values if str(v).strip()]


def query_labs(patient_record: dict) -> dict:
    """
    Extract and normalize lab results from a patient record.
    Handles schema drift — keys may be renamed (e.g. troponin_i → TROP).
    """
    try:
        if not isinstance(patient_record, dict):
            patient_record = {}

        lab_results = patient_record.get("lab_results", {})
        if not isinstance(lab_results, dict):
            lab_results = {}

        standard_to_alias = {
            "troponin": ("troponin_i", "TROP"),
            "bnp": ("bnp", "BNP"),
            "creatinine": ("creatinine", "Cr"),
            "glucose": ("glucose", "GLU"),
            "wbc": ("wbc", "WBC"),
            "hemoglobin": ("hemoglobin", "HGB"),
        }

        normalized: Dict[str, Optional[float]] = {
            "troponin": None,
            "bnp": None,
            "creatinine": None,
            "glucose": None,
            "wbc": None,
            "hemoglobin": None,
        }
        raw_keys_found: List[str] = []
        drift_detected = False

        for normalized_key, (standard_key, drift_key) in standard_to_alias.items():
            if standard_key in lab_results:
                normalized[normalized_key] = _as_float(lab_results.get(standard_key))
                raw_keys_found.append(standard_key)
                continue

            if drift_key in lab_results:
                normalized[normalized_key] = _as_float(lab_results.get(drift_key))
                raw_keys_found.append(drift_key)
                drift_detected = True

        return {
            "troponin": normalized["troponin"],
            "bnp": normalized["bnp"],
            "creatinine": normalized["creatinine"],
            "glucose": normalized["glucose"],
            "wbc": normalized["wbc"],
            "hemoglobin": normalized["hemoglobin"],
            "raw_keys_found": raw_keys_found,
            "drift_detected": drift_detected,
        }
    except Exception as e:
        return {
            "troponin": None,
            "bnp": None,
            "creatinine": None,
            "glucose": None,
            "wbc": None,
            "hemoglobin": None,
            "raw_keys_found": [],
            "drift_detected": False,
            "error": str(e),
        }


def check_allergies(patient_record: dict, drug_name: str) -> dict:
    """
    Check whether a drug is safe given a patient's known allergies
    and explicit unsafe_drugs list.
    """
    try:
        if not isinstance(patient_record, dict):
            patient_record = {}

        drug = str(drug_name or "").strip()
        drug_norm = drug.lower()

        known_allergies = _normalized_list(patient_record.get("known_allergies"))
        unsafe_drugs = _normalized_list(patient_record.get("unsafe_drugs"))
        safe_drugs = _normalized_list(patient_record.get("safe_drugs"))

        allergy_conflict = bool(drug_norm) and any(drug_norm in allergy for allergy in known_allergies)
        unsafe_listed = drug_norm in unsafe_drugs if drug_norm else False
        safe_listed = drug_norm in safe_drugs if drug_norm else False

        if allergy_conflict or unsafe_listed:
            verdict = "unsafe"
            reason = (
                f"{drug or 'Drug'} is flagged unsafe due to allergy conflict and/or unsafe_drugs listing."
            )
        elif safe_listed:
            verdict = "safe"
            reason = f"{drug or 'Drug'} appears in safe_drugs and no allergy/unsafe conflict was found."
        else:
            verdict = "unknown"
            reason = f"{drug or 'Drug'} is not explicitly safe and no explicit conflict was found."

        return {
            "drug": drug,
            "allergy_conflict": allergy_conflict,
            "unsafe_listed": unsafe_listed,
            "safe_listed": safe_listed,
            "verdict": verdict,
            "reason": reason,
        }
    except Exception as e:
        return {
            "drug": str(drug_name or ""),
            "allergy_conflict": False,
            "unsafe_listed": False,
            "safe_listed": False,
            "verdict": "unknown",
            "reason": f"Failed to evaluate allergy safety: {e}",
        }


def drug_interactions(drug_name: str, current_medications: list[str]) -> dict:
    """
    Check for known interactions between a proposed drug and
    a patient's current medications using the emergency_drugs.json DB.
    """
    try:
        drug_db = _load_emergency_drugs()
        drug = str(drug_name or "").strip()

        meds = current_medications if isinstance(current_medications, list) else []
        med_pairs = [(str(med).strip(), str(med).strip().lower()) for med in meds if str(med).strip()]

        if drug not in drug_db:
            return {
                "drug": drug,
                "interactions_found": ["Drug not in database"],
                "conflict_medications": [],
                "has_conflict": False,
                "risk_level": "none",
            }

        interaction_terms_raw = drug_db.get(drug, {}).get("interactions", [])
        interaction_terms = [str(term).strip().lower() for term in interaction_terms_raw if str(term).strip()]

        interactions_found: List[str] = []
        conflict_medications: List[str] = []

        for interaction_term in interaction_terms:
            for med_original, med_norm in med_pairs:
                if interaction_term in med_norm or med_norm in interaction_term:
                    interactions_found.append(interaction_term)
                    conflict_medications.append(med_original)

        interactions_found = sorted(set(interactions_found))
        conflict_medications = sorted(set(conflict_medications))

        conflict_count = len(conflict_medications)
        if conflict_count == 0:
            risk_level = "none"
        elif conflict_count == 1:
            risk_level = "moderate"
        else:
            risk_level = "high"

        return {
            "drug": drug,
            "interactions_found": interactions_found,
            "conflict_medications": conflict_medications,
            "has_conflict": conflict_count > 0,
            "risk_level": risk_level,
        }
    except Exception as e:
        return {
            "drug": str(drug_name or ""),
            "interactions_found": [f"Error: {e}"],
            "conflict_medications": [],
            "has_conflict": False,
            "risk_level": "none",
        }


def icd_lookup(code_or_name: str) -> dict:
    """
    Look up an ICD-10 code or condition name in the local database.
    """
    try:
        query = str(code_or_name or "").strip()
        query_norm = query.lower()
        icd_db = _load_icd10_conditions()

        if query in icd_db:
            hit = icd_db[query]
            return {
                "code": query,
                "name": str(hit.get("name")) if isinstance(hit, dict) else None,
                "category": str(hit.get("category")) if isinstance(hit, dict) else None,
                "found": True,
                "match_type": "exact_code",
            }

        for code, meta in icd_db.items():
            if not isinstance(meta, dict):
                continue
            name = str(meta.get("name", ""))
            if name and name.lower() == query_norm:
                return {
                    "code": code,
                    "name": name,
                    "category": str(meta.get("category")) if meta.get("category") is not None else None,
                    "found": True,
                    "match_type": "exact_name",
                }

        for code, meta in icd_db.items():
            if not isinstance(meta, dict):
                continue
            name = str(meta.get("name", ""))
            if query_norm and query_norm in name.lower():
                return {
                    "code": code,
                    "name": name,
                    "category": str(meta.get("category")) if meta.get("category") is not None else None,
                    "found": True,
                    "match_type": "partial_name",
                }

        return {
            "code": None,
            "name": None,
            "category": None,
            "found": False,
            "match_type": "not_found",
        }
    except Exception:
        return {
            "code": None,
            "name": None,
            "category": None,
            "found": False,
            "match_type": "not_found",
        }


def dose_check(drug_name: str, dose_mg: float) -> dict:
    """
    Validate a proposed drug dosage against the emergency drug database.
    """
    try:
        drug_db = _load_emergency_drugs()
        drug = str(drug_name or "").strip()
        proposed_dose = _as_float(dose_mg)

        if proposed_dose is None:
            proposed_dose = 0.0

        if drug not in drug_db:
            return {
                "drug": drug,
                "proposed_dose_mg": proposed_dose,
                "min_dose_mg": None,
                "max_dose_mg": None,
                "in_range": False,
                "verdict": "unknown_drug",
                "safe_midpoint_mg": None,
                "reason": f"{drug or 'Drug'} was not found in emergency drug database.",
            }

        info = drug_db.get(drug, {})
        min_dose = _as_float(info.get("min_dose_mg"))
        max_dose = _as_float(info.get("max_dose_mg"))

        if min_dose is None or max_dose is None:
            return {
                "drug": drug,
                "proposed_dose_mg": proposed_dose,
                "min_dose_mg": min_dose,
                "max_dose_mg": max_dose,
                "in_range": False,
                "verdict": "unknown_drug",
                "safe_midpoint_mg": None,
                "reason": f"{drug} dose range is incomplete in database.",
            }

        midpoint = (min_dose + max_dose) / 2.0

        if proposed_dose < min_dose:
            verdict = "too_low"
            in_range = False
            reason = f"Proposed dose {proposed_dose}mg is below minimum {min_dose}mg for {drug}."
        elif proposed_dose > max_dose:
            verdict = "too_high"
            in_range = False
            reason = f"Proposed dose {proposed_dose}mg is above maximum {max_dose}mg for {drug}."
        else:
            verdict = "safe"
            in_range = True
            reason = f"Proposed dose {proposed_dose}mg is within safe range {min_dose}-{max_dose}mg for {drug}."

        return {
            "drug": drug,
            "proposed_dose_mg": proposed_dose,
            "min_dose_mg": min_dose,
            "max_dose_mg": max_dose,
            "in_range": in_range,
            "verdict": verdict,
            "safe_midpoint_mg": midpoint,
            "reason": reason,
        }
    except Exception as e:
        return {
            "drug": str(drug_name or ""),
            "proposed_dose_mg": float(dose_mg) if isinstance(dose_mg, (int, float)) else 0.0,
            "min_dose_mg": None,
            "max_dose_mg": None,
            "in_range": False,
            "verdict": "unknown_drug",
            "safe_midpoint_mg": None,
            "reason": f"Failed to validate dose: {e}",
        }


MCP_TOOLS = {
    "query_labs": {
        "function": query_labs,
        "description": "Extract and normalize lab results, detects schema drift",
        "input_keys": ["patient_record"],
    },
    "check_allergies": {
        "function": check_allergies,
        "description": "Check drug safety against patient allergies and unsafe list",
        "input_keys": ["patient_record", "drug_name"],
    },
    "drug_interactions": {
        "function": drug_interactions,
        "description": "Check interactions between drug and current medications",
        "input_keys": ["drug_name", "current_medications"],
    },
    "icd_lookup": {
        "function": icd_lookup,
        "description": "Look up ICD-10 code or condition name in local database",
        "input_keys": ["code_or_name"],
    },
    "dose_check": {
        "function": dose_check,
        "description": "Validate a proposed drug dose against safety range in DB",
        "input_keys": ["drug_name", "dose_mg"],
    },
}


def list_tools() -> list[str]:
    """Return names of all available MCP tools."""
    return list(MCP_TOOLS.keys())


def call_tool(tool_name: str, **kwargs) -> dict:
    """
    Dispatch a tool call by name.
    Returns {"error": "..."} if tool not found or call fails.
    """
    if tool_name not in MCP_TOOLS:
        return {"error": f"Unknown tool: {tool_name}. Available: {list_tools()}"}
    try:
        return MCP_TOOLS[tool_name]["function"](**kwargs)
    except Exception as e:
        return {"error": str(e), "tool": tool_name}


if __name__ == "__main__":
    # Quick self-test — run: python tools/mcp_tools.py

    sample_patient = {
        "lab_results": {"troponin_i": 3.8, "bnp": 220, "wbc": 9.6},
        "known_allergies": ["aspirin"],
        "safe_drugs": ["Nitroglycerin"],
        "unsafe_drugs": ["Aspirin"],
        "current_medications": ["lisinopril"],
    }

    print("=== query_labs ===")
    print(query_labs(sample_patient))

    print("\n=== check_allergies (Aspirin) ===")
    print(check_allergies(sample_patient, "Aspirin"))

    print("\n=== check_allergies (Nitroglycerin) ===")
    print(check_allergies(sample_patient, "Nitroglycerin"))

    print("\n=== drug_interactions ===")
    print(drug_interactions("Epinephrine", ["lisinopril", "metformin"]))

    print("\n=== icd_lookup ===")
    print(icd_lookup("I21.9"))
    print(icd_lookup("myocardial"))

    print("\n=== dose_check ===")
    print(dose_check("Nitroglycerin", 0.4))
    print(dose_check("Nitroglycerin", 50.0))

    print("\n=== call_tool dispatch ===")
    print(call_tool("dose_check", drug_name="Morphine", dose_mg=5.0))
    print(call_tool("unknown_tool"))

    print("\n=== list_tools ===")
    print(list_tools())
