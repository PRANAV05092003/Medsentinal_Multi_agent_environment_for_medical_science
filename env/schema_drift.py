"""
MedSentinel Schema Drift Simulation
==================================

This module simulates *schema drift* attacks by renaming keys inside the nested
`vitals` and `lab_results` dictionaries of a patient case.

Design goals (hackathon-friendly):
- Realistic: uses plausible aliases (e.g., `troponin` -> `TROP`)
- Robust: never deletes values; only renames keys and avoids collisions
- Deterministic enough for debugging: accepts an explicit seed and uses a local RNG
- Reusable: small helper functions and clean interfaces

Public API
---------
apply_schema_drift(patient, ...) -> (modified_patient, drift_occurred, changes_dict)

Where `changes_dict` has the shape:
{
  "vitals": {"old_key": "new_key", ...},
  "lab_results": {"old_key": "new_key", ...}
}
"""

from __future__ import annotations

import copy
import random
import re
from typing import Any, Dict, Mapping, MutableMapping, Tuple


# Realistic aliases that are common in ED/ICU charting or abbreviated lab panels.
# Keep this mapping intentionally small and easy to audit/extend.
VITAL_ALIASES: Dict[str, str] = {
    "hr": "HR",
    "heart_rate": "HR",
    "pulse": "PULSE",
    "rr": "RR",
    "resp_rate": "RR",
    "respiratory_rate": "RR",
    "temp_c": "TEMP_C",
    "temperature_c": "TEMP_C",
    "temp_f": "TEMP_F",
    "temperature_f": "TEMP_F",
    "temperature": "TEMP_C",
    "spo2_pct": "SpO2",
    "spo2": "SpO2",
    "o2_sat": "SpO2",
    "bp_systolic": "SBP",
    "bp_diastolic": "DBP",
    "sbp": "SBP",
    "dbp": "DBP",
    "map": "MAP",
}


LAB_ALIASES: Dict[str, str] = {
    "wbc": "WBC",
    "hb": "HGB",
    "hgb": "HGB",
    "hemoglobin": "HGB",
    "platelets": "PLT",
    "plt": "PLT",
    "sodium": "Na",
    "na": "Na",
    "potassium": "K",
    "k": "K",
    "chloride": "Cl",
    "cl": "Cl",
    "bicarbonate": "HCO3",
    "hco3": "HCO3",
    "bun": "BUN",
    "creatinine": "Cr",
    "cr": "Cr",
    "glucose": "GLU",
    "bnp": "BNP",
    "lactate": "LAC",
    "troponin": "TROP",
    "troponin_i": "TROP",
    "inr": "INR",
    "pt": "PT",
    "ptt": "aPTT",
    "aptt": "aPTT",
}


def _normalize_key(key: str) -> str:
    """
    Normalize keys for alias lookups:
    - lowercased
    - collapse inner whitespace
    - strip surrounding punctuation
    """
    k = re.sub(r"\s+", " ", key.strip().lower())
    return k.strip(" .,:;\"'()[]{}")


def _is_mapping(x: Any) -> bool:
    return isinstance(x, Mapping)


def _coerce_dict(x: Any) -> Dict[str, Any]:
    """
    Convert a Mapping-like object to a mutable dict[str, Any].
    Returns empty dict for invalid input.
    """
    if not _is_mapping(x):
        return {}
    out: Dict[str, Any] = {}
    for k, v in x.items():
        if isinstance(k, str):
            out[k] = v
    return out


def _choose_keys_to_drift(keys: Tuple[str, ...], rng: random.Random, max_renames: int) -> Tuple[str, ...]:
    """
    Choose which keys to rename.
    Selection is deterministic given the RNG seed and input key order.
    """
    if max_renames <= 0 or not keys:
        return tuple()
    if len(keys) <= max_renames:
        return keys
    return tuple(rng.sample(list(keys), k=max_renames))


def _unique_target_key(desired: str, existing_keys: Tuple[str, ...], rng: random.Random) -> str:
    """
    Ensure the target key does not collide with existing keys.
    """
    if desired not in existing_keys:
        return desired

    for i in range(2, 10):
        candidate = f"{desired}_{i}"
        if candidate not in existing_keys:
            return candidate

    for _ in range(50):
        candidate = f"{desired}_{rng.randint(10, 99)}"
        if candidate not in existing_keys:
            return candidate

    return desired


def _build_rename_map(
    original: Mapping[str, Any],
    aliases: Mapping[str, str],
    rng: random.Random,
    max_renames: int,
) -> Dict[str, str]:
    """
    Build a mapping {old_key: new_key} for keys eligible for drift.
    Keys without a known alias are left unchanged (not selected).
    """
    if not _is_mapping(original):
        return {}

    existing_keys = tuple(k for k in original.keys() if isinstance(k, str))
    eligible = tuple(k for k in existing_keys if _normalize_key(k) in aliases)
    chosen = _choose_keys_to_drift(eligible, rng=rng, max_renames=max_renames)

    rename_map: Dict[str, str] = {}
    for old in chosen:
        desired = aliases[_normalize_key(old)]
        new = _unique_target_key(desired, existing_keys=existing_keys, rng=rng)
        if new != old:
            rename_map[old] = new
    return rename_map


def _apply_key_renames(dct: MutableMapping[str, Any], rename_map: Mapping[str, str]) -> Dict[str, Any]:
    """
    Apply renames to a dict without changing its values.
    Preserves all untouched keys.
    """
    if not rename_map:
        return dict(dct)

    out: Dict[str, Any] = {}
    for k, v in dct.items():
        if not isinstance(k, str):
            continue
        out[rename_map.get(k, k)] = v
    return out


def apply_schema_drift(
    patient: Mapping[str, Any],
    *,
    seed: int = 1337,
    drift_probability: float = 0.35,
    max_key_renames_per_section: int = 2,
) -> Tuple[Dict[str, Any], bool, Dict[str, Dict[str, str]]]:
    """
    Apply schema drift to a patient record by renaming keys in `vitals` and `lab_results`.

    Returns:
    - modified_patient
    - drift_occurred
    - changes_dict
    """
    rng = random.Random(seed)
    modified: Dict[str, Any] = copy.deepcopy(dict(patient)) if _is_mapping(patient) else {}

    changes: Dict[str, Dict[str, str]] = {"vitals": {}, "lab_results": {}}

    if drift_probability <= 0.0:
        return modified, False, changes
    do_drift = True if drift_probability >= 1.0 else (rng.random() < drift_probability)
    if not do_drift:
        return modified, False, changes

    vitals = _coerce_dict(modified.get("vitals"))
    labs = _coerce_dict(modified.get("lab_results"))

    vitals_rename = _build_rename_map(vitals, aliases=VITAL_ALIASES, rng=rng, max_renames=max_key_renames_per_section)
    labs_rename = _build_rename_map(labs, aliases=LAB_ALIASES, rng=rng, max_renames=max_key_renames_per_section)

    if isinstance(modified.get("vitals"), Mapping):
        modified["vitals"] = _apply_key_renames(vitals, vitals_rename)
        changes["vitals"] = dict(vitals_rename)
    if isinstance(modified.get("lab_results"), Mapping):
        modified["lab_results"] = _apply_key_renames(labs, labs_rename)
        changes["lab_results"] = dict(labs_rename)

    drift_occurred = bool(changes["vitals"] or changes["lab_results"])
    return modified, drift_occurred, changes

