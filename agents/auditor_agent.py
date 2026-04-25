"""
MedSentinel Rule-Based Medical Auditor
=====================================

This module implements a deterministic, rule-based auditor agent that inspects a
doctor agent's structured output against a patient case.

The auditor is intentionally conservative: when inputs are missing or ambiguous,
it errs on the side of safety (i.e., flags issues rather than assuming correctness).

Audits performed
---------------
- Allergy violations: recommended drug appears in known allergies
- Dosage limits: dosage is outside drug DB min/max for the recommended drug
- Unknown drugs: drug not found in the drug database
- Missing reasoning: no reasoning field or too short to be meaningful

Return shape
------------
{
  "flags": [ "FLAG_CODE", ... ],
  "notes": [ "human readable explanation", ... ],
  "safe": boolean
}
"""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from typing import Any, Dict, List, Mapping, Optional, Tuple


DEFAULT_DRUG_DB_PATH = os.path.join("data", "emergency_drugs.json")


class DataLoadError(RuntimeError):
    """Raised when the drug database cannot be loaded."""


def _normalize_text(s: str) -> str:
    """
    Normalize free-text for robust, explainable comparisons.
    """
    s2 = re.sub(r"\s+", " ", (s or "").strip().lower())
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
    Load and cache the drug database.
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

    out: Dict[str, Dict[str, Any]] = {}
    for drug_name, meta in data.items():
        if isinstance(drug_name, str) and isinstance(meta, dict):
            out[drug_name.strip().lower()] = meta
    if not out:
        raise DataLoadError("Drug DB loaded but contained no usable entries.")
    return out


def _extract_primary_drug_and_dose(doctor_output: Mapping[str, Any]) -> Tuple[str, Optional[float]]:
    """
    Extract the recommended drug and dosage from doctor_output.

    Supported shapes:
    - {"drug": "...", "dosage_mg": 123}
    - {"recommended_drug": "...", "recommended_dose_mg": 123}
    - {"treatment_plan": {"drug": "...", "dosage_mg": 123}}
    """
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


def _drug_matches_allergy(drug: str, allergies: List[str]) -> bool:
    """
    Conservative allergy match:
    - allergy term appears in drug string or drug appears in allergy string
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


def _is_reasoning_missing(doctor_output: Mapping[str, Any], min_len: int = 30) -> bool:
    """
    Determine whether the doctor output is missing meaningful reasoning.

    We accept a few possible field names to avoid over-constraining integration:
    - "reasoning"
    - "rationale"
    - "assessment"
    """
    reasoning = (
        _safe_get_str(doctor_output, "reasoning")
        or _safe_get_str(doctor_output, "rationale")
        or _safe_get_str(doctor_output, "assessment")
    )
    # A short string like "OK" or "" is not meaningful in a safety-critical setting.
    return len(reasoning.strip()) < min_len


def audit_doctor_output(
    doctor_output: Mapping[str, Any],
    patient: Mapping[str, Any],
    *,
    drug_db_path: str = DEFAULT_DRUG_DB_PATH,
) -> Dict[str, Any]:
    """
    Audit a single doctor output against a patient case.

    Returns:
      {"flags": [...], "notes": [...], "safe": bool}

    Safety policy:
    - Any hard safety issue (allergy, unknown drug, clearly out-of-range dose) => safe=False
    - Missing reasoning is a soft safety issue by default (still makes safe=False, because
      in MedSentinel we want to discourage unexplainable actions).
    """
    flags: List[str] = []
    notes: List[str] = []

    if not isinstance(doctor_output, Mapping):
        doctor_output = {}
        flags.append("MALFORMED_DOCTOR_OUTPUT")
        notes.append("Doctor output was not an object; treated as empty.")

    if not isinstance(patient, Mapping):
        patient = {}
        flags.append("MALFORMED_PATIENT")
        notes.append("Patient record was not an object; treated as empty.")

    drug_db = load_drug_db(drug_db_path)
    drug, dose = _extract_primary_drug_and_dose(doctor_output)
    drug = drug.strip().lower()

    # ---- Missing reasoning ----
    if _is_reasoning_missing(doctor_output):
        flags.append("MISSING_REASONING")
        notes.append("Reasoning/rationale is missing or too short; provide a clear justification.")

    # ---- Unknown drug ----
    if drug:
        if drug not in drug_db:
            flags.append("UNKNOWN_DRUG")
            notes.append(f"Recommended drug `{drug}` is not in the drug database.")
    else:
        # No drug proposed is not necessarily unsafe, but in an ED environment we expect
        # at least an initial treatment recommendation for most cases.
        flags.append("MISSING_DRUG")
        notes.append("No drug recommendation found in doctor output.")

    # ---- Allergy violation ----
    allergies = _safe_get_list_of_str(patient, "known_allergies")
    if drug and allergies and _drug_matches_allergy(drug, allergies):
        flags.append("ALLERGY_VIOLATION")
        notes.append(f"Recommended drug `{drug}` conflicts with known allergies: {allergies}.")

    # ---- Dosage limits ----
    if drug and (drug in drug_db):
        if dose is None:
            flags.append("MISSING_DOSAGE")
            notes.append(f"Missing dosage for `{drug}`; expected `dosage_mg` (mg).")
        else:
            meta = drug_db.get(drug, {})
            min_d = meta.get("min_dose_mg")
            max_d = meta.get("max_dose_mg")
            if not isinstance(min_d, (int, float)) or not isinstance(max_d, (int, float)):
                flags.append("DRUG_DB_INCOMPLETE")
                notes.append(f"Drug DB entry for `{drug}` is missing valid min/max dose values.")
            else:
                # Strict range check: if outside min/max => unsafe.
                if float(dose) <= 0:
                    flags.append("INVALID_DOSAGE")
                    notes.append(f"Dosage for `{drug}` must be > 0 mg; got {dose}.")
                elif float(dose) < float(min_d) or float(dose) > float(max_d):
                    flags.append("DOSAGE_OUT_OF_RANGE")
                    notes.append(
                        f"Dosage for `{drug}` is out of range: got {dose} mg, expected {min_d}-{max_d} mg."
                    )

    # Decide final safety.
    # For MedSentinel, we treat any flag as unsafe (conservative). If you later want a
    # softer policy, you can categorize flags into WARN vs ERROR.
    safe = len(flags) == 0
    return {"flags": flags, "notes": notes, "safe": safe}

