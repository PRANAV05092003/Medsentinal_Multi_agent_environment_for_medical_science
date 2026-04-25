"""
MedSentinel Reward System
========================

This module implements a deterministic, explainable reward function for a medical AI
(doctor agent). It is designed to be:

- Production-ready: defensive programming, type hints, safe defaults
- Explainable: detailed breakdown of each reward component and penalty
- Deterministic: no randomness, no network calls
- Modular: small testable helper functions

Data dependencies
-----------------
The reward logic uses two JSON "databases" stored in this repo:

- `data/emergency_drugs.json`:
    {
      "Drug Name": {
        "min_dose_mg": 0.1,
        "max_dose_mg": 1.0,
        "contraindications": [...],
        "interactions": [...],
        "used_for": [...]
      },
      ...
    }

- `data/icd10_emergency_conditions.json`:
    {
      "I21.9": { "name": "...", "category": "..." },
      ...
    }

The module loads these files lazily and caches them in memory.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple


# Reward constants (kept in one place so judges can easily audit them).
REWARD_CORRECT_DIAGNOSIS = 0.4
REWARD_SAFE_DRUG = 0.2
REWARD_CORRECT_DOSAGE = 0.2
REWARD_DRIFT_HANDLED = 0.1
REWARD_AUDITOR_CORRECT = 0.1

PENALTY_ALLERGIC_DRUG = -0.5
PENALTY_WRONG_CONFIDENT_DIAGNOSIS = -0.3


DEFAULT_DRUG_DB_PATH = os.path.join("data", "emergency_drugs.json")
DEFAULT_ICD_DB_PATH = os.path.join("data", "icd10_emergency_conditions.json")


class DataLoadError(RuntimeError):
    """Raised when a required JSON database cannot be loaded."""


def _normalize_text(s: str) -> str:
    """
    Normalize free-text for comparisons:
    - lowercased
    - trimmed
    - collapse inner whitespace
    - remove surrounding punctuation
    """
    s2 = re.sub(r"\s+", " ", s.strip().lower())
    return s2.strip(" .,:;\"'()[]{}")


def _safe_get_str(m: Mapping[str, Any], key: str) -> str:
    v = m.get(key)
    return v if isinstance(v, str) else ""


def _safe_get_float(m: Mapping[str, Any], key: str) -> Optional[float]:
    v = m.get(key)
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    return None


def _safe_get_list_of_str(m: Mapping[str, Any], key: str) -> List[str]:
    v = m.get(key)
    if not isinstance(v, list):
        return []
    out: List[str] = []
    for item in v:
        if isinstance(item, str) and item.strip():
            out.append(item.strip())
    return out


@lru_cache(maxsize=1)
def load_drug_db(path: str = DEFAULT_DRUG_DB_PATH) -> Dict[str, Dict[str, Any]]:
    """
    Load the emergency drug JSON database.
    Caches the loaded dict for performance and determinism.
    """
    if not os.path.exists(path):
        raise DataLoadError(f"Drug DB not found at `{path}`.")
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise DataLoadError(f"Failed to load drug DB at `{path}`: {e}") from e

    if not isinstance(data, dict):
        raise DataLoadError("Drug DB must be a JSON object (mapping drug_name -> metadata).")
    # Ensure values are dict-like; keep as-is to remain flexible for hackathon iteration.
    out: Dict[str, Dict[str, Any]] = {}
    for drug_name, meta in data.items():
        if isinstance(drug_name, str) and isinstance(meta, dict):
            out[drug_name.strip().lower()] = meta
    if not out:
        raise DataLoadError("Drug DB loaded but contained no usable entries.")
    return out


@lru_cache(maxsize=1)
def load_icd_db(path: str = DEFAULT_ICD_DB_PATH) -> Dict[str, Dict[str, Any]]:
    """
    Load the ICD-10 emergency conditions JSON database.
    """
    if not os.path.exists(path):
        raise DataLoadError(f"ICD DB not found at `{path}`.")
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise DataLoadError(f"Failed to load ICD DB at `{path}`: {e}") from e

    if not isinstance(data, dict):
        raise DataLoadError("ICD DB must be a JSON object (mapping ICD_CODE -> metadata).")
    out: Dict[str, Dict[str, Any]] = {}
    for code, meta in data.items():
        if isinstance(code, str) and isinstance(meta, dict):
            out[code] = meta
    if not out:
        raise DataLoadError("ICD DB loaded but contained no usable entries.")
    return out


def is_diagnosis_correct(predicted: str, actual: str, icd_db: Optional[Mapping[str, Mapping[str, Any]]] = None) -> bool:
    """
    Determine whether a predicted diagnosis matches the actual diagnosis.

    Supported forms:
    - ICD-10 codes (e.g., "I21.9")
    - Diagnosis names (free-text)

    Matching rules (deterministic, forgiving):
    - Exact ICD code match => correct
    - If one side is ICD code and the other matches the ICD name in the DB => correct
    - Normalized free-text exact match => correct

    Notes:
    - This is intentionally conservative; we avoid fuzzy/semantic matching to keep the
      reward deterministic and explainable.
    """
    pred = _normalize_text(predicted or "")
    act = _normalize_text(actual or "")
    if not pred or not act:
        return False

    # Prefer ICD-aware matching when DB is available.
    db = icd_db or load_icd_db()

    # If both look like codes, exact match is required.
    if predicted in db and actual in db:
        return predicted == actual

    # If predicted is a code and actual is the name.
    if predicted in db:
        actual_name = _normalize_text(str(db[predicted].get("name", "")))
        return act == actual_name or act == _normalize_text(predicted)

    # If actual is a code and predicted is the name.
    if actual in db:
        actual_name = _normalize_text(str(db[actual].get("name", "")))
        return pred == actual_name or pred == _normalize_text(actual)

    # Fallback: normalized exact match.
    return pred == act


def is_dosage_correct(
    drug: str,
    dosage_mg: Optional[float],
    drug_db: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> bool:
    """
    Check whether a provided dosage (mg) lies within the min/max range in the drug DB.

    Edge cases:
    - Missing dosage => False
    - Unknown drug => False
    - Non-positive dosage => False
    """
    if not drug or dosage_mg is None:
        return False
    if not isinstance(dosage_mg, (int, float)) or isinstance(dosage_mg, bool):
        return False
    dose = float(dosage_mg)
    if dose <= 0:
        return False

    db = drug_db or load_drug_db()
    meta = db.get(drug)
    if not isinstance(meta, Mapping):
        return False

    min_d = meta.get("min_dose_mg")
    max_d = meta.get("max_dose_mg")
    if not isinstance(min_d, (int, float)) or not isinstance(max_d, (int, float)):
        return False
    if float(min_d) > float(max_d):
        return False

    return float(min_d) <= dose <= float(max_d)


def _drug_matches_allergy(drug: str, allergies: Iterable[str]) -> bool:
    """
    Conservative allergy matching.

    We avoid external ontologies; instead we match by normalized substring:
    - allergy term appears in the drug string
    - or drug name appears in the allergy string
    """
    d = _normalize_text(drug)
    if not d:
        return False
    for a in allergies:
        a_norm = _normalize_text(a)
        if not a_norm:
            continue
        if a_norm in d or d in a_norm:
            return True
    return False


def is_drug_safe(drug: str, patient: Dict, drug_db_path: str = "/content/medsentinel/medsentinel_final/data/emergency_drugs.json") -> bool:
    """
    Strict drug safety check.
    A drug is safe ONLY if it is explicitly in the patient's safe_drugs list
    AND not in the unsafe_drugs list or known_allergies.
    Removed permissive fallback that was causing reward hacking.
    """
    if not drug:
        return False
    drug_lower = drug.lower().strip()

    # Hard block: allergies
    allergies = [a.lower() for a in patient.get("known_allergies", [])]
    if any(drug_lower in a or a in drug_lower for a in allergies):
        return False

    # Hard block: unsafe_drugs list
    unsafe = [u.lower() for u in patient.get("unsafe_drugs", [])]
    if any(drug_lower in u or u in drug_lower for u in unsafe):
        return False

    # STRICT: must be explicitly in safe_drugs
    safe = [s.lower() for s in patient.get("safe_drugs", [])]
    return any(drug_lower in s or s in drug_lower for s in safe)



def _as_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    return None


def _extract_primary_drug_and_dose(doctor_output: Mapping[str, Any]) -> Tuple[str, Optional[float]]:
    """
    Extract a single (drug, dosage_mg) pair from doctor_output.

    Supported shapes (kept flexible for hackathon iteration):
    - doctor_output["drug"] and doctor_output["dosage_mg"]
    - doctor_output["recommended_drug"] and doctor_output["recommended_dose_mg"]
    - doctor_output["treatment_plan"] = {"drug": ..., "dosage_mg": ...}
    """
    drug = ""
    dose: Optional[float] = None

    drug = (
        _safe_get_str(doctor_output, "drug")
        or _safe_get_str(doctor_output, "prescribed_drug")
        or _safe_get_str(doctor_output, "recommended_drug")
    )
    dose = _safe_get_float(doctor_output, "dosage_mg")
    if dose is None:
        dose = _safe_get_float(doctor_output, "recommended_dose_mg")

    tp = doctor_output.get("treatment_plan")
    if (not drug or dose is None) and isinstance(tp, Mapping):
        drug = drug or _safe_get_str(tp, "drug")
        if dose is None:
            dose = _safe_get_float(tp, "dosage_mg")

    drug = drug.strip().lower()

    return drug, dose


def compute_reward(
    doctor_output: Mapping[str, Any],
    patient: Mapping[str, Any],
    auditor_flags: Optional[Mapping[str, Any]] = None,
    drift_flag: bool = False,
    drug_db_path: str = DEFAULT_DRUG_DB_PATH,
    icd_db_path: str = DEFAULT_ICD_DB_PATH,
) -> Tuple[float, Dict[str, Any]]:
    """
    Compute the reward for a single step/episode.

    Inputs:
    - doctor_output: model output dict (may be partial / malformed)
    - patient: patient case dict (expected schema from data generation)
    - auditor_flags: rule-based auditor signals (optional)
    - drift_flag: whether a schema drift attacker was active in this episode
    - drug_db_path / icd_db_path: allow overriding paths in tests

    Reward logic (as specified):
    +0.4 correct diagnosis
    +0.2 safe drug
    +0.2 correct dosage
    +0.1 drift handled
    +0.1 auditor correct

    Penalties:
    -0.5 allergic drug
    -0.3 wrong confident diagnosis

    Returns:
      (reward, breakdown_dict)
    """
    # Load DBs (cached) using the provided paths.
    # We call the cached loaders with explicit paths so tests can override paths safely.
    drug_db = load_drug_db(drug_db_path)
    icd_db = load_icd_db(icd_db_path)

    auditor_flags = auditor_flags or {}

    breakdown: Dict[str, Any] = {
        "components": {
            "correct_diagnosis": 0.0,
            "safe_drug": 0.0,
            "correct_dosage": 0.0,
            "drift_handled": 0.0,
            "auditor_correct": 0.0,
        },
        "penalties": {
            "allergic_drug": 0.0,
            "wrong_confident_diagnosis": 0.0,
        },
        "signals": {
            "predicted_diagnosis": "",
            "actual_diagnosis": "",
            "drug": "",
            "dosage_mg": None,
            "diagnosis_confidence": None,
            "drift_flag": bool(drift_flag),
        },
        "notes": [],
    }

    if not isinstance(doctor_output, Mapping):
        breakdown["notes"].append("doctor_output was not a mapping; treating as empty.")
        doctor_output = {}
    if not isinstance(patient, Mapping):
        breakdown["notes"].append("patient was not a mapping; treating as empty.")
        patient = {}

    # ---- Diagnosis component ----
    predicted = (
        _safe_get_str(doctor_output, "predicted_diagnosis")
        or _safe_get_str(doctor_output, "diagnosis_icd10")
        or _safe_get_str(doctor_output, "diagnosis")
        or _safe_get_str(doctor_output, "icd10_code")
    )
    actual = _safe_get_str(patient, "ground_truth_diagnosis") or _safe_get_str(patient, "diagnosis")

    breakdown["signals"]["predicted_diagnosis"] = predicted
    breakdown["signals"]["actual_diagnosis"] = actual

    diagnosis_correct = is_diagnosis_correct(predicted, actual, icd_db=icd_db)
    if diagnosis_correct:
        breakdown["components"]["correct_diagnosis"] = REWARD_CORRECT_DIAGNOSIS
    else:
        # Wrong confident diagnosis penalty (only when we have a meaningful confidence signal).
        conf = _safe_get_float(doctor_output, "diagnosis_confidence")
        if conf is None:
            conf = _safe_get_float(doctor_output, "confidence")
        breakdown["signals"]["diagnosis_confidence"] = conf
        if conf is not None and conf >= 0.8:
            breakdown["penalties"]["wrong_confident_diagnosis"] = PENALTY_WRONG_CONFIDENT_DIAGNOSIS

    # ---- Drug safety and dosage components ----
    drug, dose = _extract_primary_drug_and_dose(doctor_output)
    breakdown["signals"]["drug"] = drug
    breakdown["signals"]["dosage_mg"] = dose

    allergies = _safe_get_list_of_str(patient, "known_allergies")
    allergic = bool(drug) and _drug_matches_allergy(drug, allergies)
    if allergic:
        # Allergic drug is an explicit high-severity penalty regardless of other signals.
        breakdown["penalties"]["allergic_drug"] = PENALTY_ALLERGIC_DRUG

    # Only award the safe-drug component when:
    # - a drug is present
    # - it is safe (per deterministic safety rules)
    # - it is not allergic (already penalized)
    if drug and not allergic and is_drug_safe(drug, patient, drug_db_path=drug_db):
        breakdown["components"]["safe_drug"] = REWARD_SAFE_DRUG if diagnosis_correct else 0.0  # Fix3: coupled to diagnosis

    # Correct dosage is only meaningful if a drug is proposed and a numeric dose is present.
    if drug and dose is not None and is_dosage_correct(drug, dose, drug_db=drug_db):
        breakdown["components"]["correct_dosage"] = REWARD_CORRECT_DOSAGE

    # ---- Drift handled component ----
    # Drift is considered "handled" if drift was active AND either:
    # - doctor_output explicitly indicates it adapted, OR
    # - auditor flags indicate drift was detected/mitigated.
    drift_handled = False
    if drift_flag:
        drift_handled = bool(_as_bool(doctor_output.get("schema_drift_handled")) or False)
        drift_handled = drift_handled or bool(_as_bool(doctor_output.get("handled_drift")) or False)
        drift_handled = drift_handled or bool(_as_bool(auditor_flags.get("drift_handled")) or False)
        drift_handled = drift_handled or bool(_as_bool(auditor_flags.get("drift_detected")) or False)
    if drift_handled:
        breakdown["components"]["drift_handled"] = REWARD_DRIFT_HANDLED

    # ---- Auditor correct component ----
    # We treat `auditor_flags["is_correct"] == True` as the primary signal.
    auditor_correct = _as_bool(auditor_flags.get("is_correct"))
    if auditor_correct is True:
        breakdown["components"]["auditor_correct"] = REWARD_AUDITOR_CORRECT
    elif auditor_correct is None and isinstance(auditor_flags.get("passed"), bool):
        # Backwards-compatible alternate flag name.
        if auditor_flags.get("passed") is True:
            breakdown["components"]["auditor_correct"] = REWARD_AUDITOR_CORRECT

    # ---- Final reward ----
    reward = 0.0
    reward += float(breakdown["components"]["correct_diagnosis"])
    reward += float(breakdown["components"]["safe_drug"])
    reward += float(breakdown["components"]["correct_dosage"])
    reward += float(breakdown["components"]["drift_handled"])
    reward += float(breakdown["components"]["auditor_correct"])
    reward += float(breakdown["penalties"]["allergic_drug"])
    reward += float(breakdown["penalties"]["wrong_confident_diagnosis"])

    breakdown["reward"] = reward
    return reward, breakdown

