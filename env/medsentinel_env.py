"""
MedSentinel Gym-Style Environment
================================

This module defines a gym-style environment class, `MedSentinelEnv`, for running
single-step medical decision episodes:

- `reset()` selects a patient case and applies schema drift to vitals/labs
- `step(doctor_output)` audits the output and computes deterministic reward

Key design goals:
- Clean architecture: separate dataset loading, splitting, drift, auditing, reward
- Fully modular: paths and hyperparameters are configurable (no hardcoding)
- Deterministic enough for debugging: controlled via `seed`

Return signature (gym-style)
----------------------------
- reset(...) -> drifted_patient_dict
- step(doctor_output) -> (reward: float, done: bool, info: dict)

Note:
This module does NOT require `gym` / `gymnasium` to be installed.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

from agents.auditor_agent import audit_doctor_output
from agents.clinical_verification_layer import ClinicalVerificationLayer
from env.reward_system import compute_reward
from env.schema_drift import apply_schema_drift


class DatasetError(RuntimeError):
    """Raised when the patient dataset cannot be loaded or is invalid."""


def load_patient_dataset(path: str) -> List[Dict[str, Any]]:
    """
    Load a patient dataset JSON file.
    """
    if not os.path.exists(path):
        raise DatasetError(f"Patient dataset not found at `{path}`.")
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise DatasetError(f"Failed to load patient dataset `{path}`: {e}") from e

    if not isinstance(data, list):
        raise DatasetError("Patient dataset must be a JSON array.")

    usable: List[Dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        pid = item.get("patient_id")
        if not isinstance(pid, str) or not pid.strip():
            continue
        usable.append(item)

    if not usable:
        raise DatasetError("Patient dataset loaded but contained no usable patient cases.")
    return usable


def train_test_split_indices(n: int, *, test_fraction: float = 0.2, seed: int = 1337) -> Tuple[List[int], List[int]]:
    """
    Deterministic train/test split by indices.
    """
    if n <= 0:
        return [], []
    if not (0.0 < test_fraction < 1.0):
        raise ValueError("test_fraction must be between 0 and 1 (exclusive).")

    rng = random.Random(seed)
    idx = list(range(n))
    rng.shuffle(idx)

    test_size = max(1, int(round(n * test_fraction)))
    test_idx = idx[:test_size]
    train_idx = idx[test_size:] or idx[: max(0, n - test_size)]
    return train_idx, test_idx


@dataclass
class EnvConfig:
    patient_dataset_path: str
    test_fraction: float = 0.2
    seed: int = 1337
    drift_probability: float = 0.35
    max_key_renames_per_section: int = 2
    mode: str = "train"  # "train" or "test"

    drug_db_path: str = os.path.join("data", "emergency_drugs.json")
    icd_db_path: str = os.path.join("data", "icd10_emergency_conditions.json")

    # CVL: disable during GRPO training for speed, enable for demo/production
    enable_cvl: bool = False


class MedSentinelEnv:
    """
    Single-step environment:
    - `reset()` returns a drifted patient record (the observation)
    - `step(doctor_output)` evaluates and ends the episode (done=True)
    """

    def __init__(self, config: EnvConfig) -> None:
        if config.mode not in ("train", "test"):
            raise ValueError("config.mode must be 'train' or 'test'.")

        self.config = config
        self._rng = random.Random(config.seed)
        # Clinical Verification Layer — lazy init on first use
        self._cvl: Optional[ClinicalVerificationLayer] = None

        self._dataset = load_patient_dataset(config.patient_dataset_path)
        train_idx, test_idx = train_test_split_indices(
            len(self._dataset),
            test_fraction=config.test_fraction,
            seed=config.seed,
        )
        self._split = {"train": train_idx, "test": test_idx}

        self._episode_idx = 0
        self._cursor = {"train": 0, "test": 0}
        self._current_patient_original: Optional[Dict[str, Any]] = None
        self._current_patient_observed: Optional[Dict[str, Any]] = None
        self._current_drift_flag: bool = False
        self._current_drift_changes: Dict[str, Dict[str, str]] = {"vitals": {}, "lab_results": {}}

    @property
    def mode(self) -> str:
        return self.config.mode

    @property
    def current_patient_original(self) -> Optional[Dict[str, Any]]:
        """
        Original (non-drifted) patient dict for the current episode.
        """
        return dict(self._current_patient_original) if isinstance(self._current_patient_original, dict) else None

    @property
    def current_patient_observed(self) -> Optional[Dict[str, Any]]:
        """
        Observed (possibly drifted) patient dict for the current episode.
        """
        return dict(self._current_patient_observed) if isinstance(self._current_patient_observed, dict) else None

    @property
    def current_drift_info(self) -> Dict[str, Any]:
        """
        Drift metadata for the current episode.
        """
        return {
            "drift_occurred": bool(self._current_drift_flag),
            "drift_changes": dict(self._current_drift_changes),
        }

    def set_mode(self, mode: str) -> None:
        if mode not in ("train", "test"):
            raise ValueError("mode must be 'train' or 'test'.")
        self.config.mode = mode

    def _next_patient_index(self, mode: str) -> int:
        indices = self._split[mode]
        if not indices:
            return self._rng.randrange(0, len(self._dataset))

        c = self._cursor[mode]
        idx = indices[c % len(indices)]
        self._cursor[mode] = c + 1
        return idx

    def reset(self, *, mode: Optional[str] = None) -> Dict[str, Any]:
        """
        Start a new episode and return the observed patient record (possibly drifted).
        """
        if mode is not None:
            self.set_mode(mode)

        dataset_idx = self._next_patient_index(self.config.mode)
        patient = self._dataset[dataset_idx]

        self._current_patient_original = dict(patient)

        drift_seed = self.config.seed + self._episode_idx
        observed, drift_flag, changes = apply_schema_drift(
            patient,
            seed=drift_seed,
            drift_probability=self.config.drift_probability,
            max_key_renames_per_section=self.config.max_key_renames_per_section,
        )

        self._current_patient_observed = observed
        self._current_drift_flag = drift_flag
        self._current_drift_changes = changes

        self._episode_idx += 1
        return observed

    def _get_cvl(self) -> ClinicalVerificationLayer:
        """Lazy-init the Clinical Verification Layer."""
        if self._cvl is None:
            self._cvl = ClinicalVerificationLayer()
        return self._cvl

    def step(self, doctor_output: Mapping[str, Any]) -> Tuple[float, bool, Dict[str, Any]]:
        """
        Evaluate one action (doctor_output) and end the episode.

        Pipeline:
          1. Doctor output arrives
          2. Auditor checks raw doctor output (rule-based safety flags)
          3. [IF CVL ENABLED] Clinical Verification Layer refines the output
             using Claude API as a senior clinician reviewer
             NOTE: CVL does NOT affect reward — reward uses raw doctor output
          4. Reward computed on raw doctor output (not CVL-refined)
          5. Info dict includes both raw and CVL-verified output for transparency
        """
        if self._current_patient_observed is None:
            raise RuntimeError("Environment must be reset() before step().")

        patient_observed = self._current_patient_observed
        patient_original = self._current_patient_original or patient_observed

        # ── Step 1: Raw doctor output (used for reward) ──────────────────────
        raw_doctor_output = doctor_output if isinstance(doctor_output, Mapping) else {}

        # ── Step 2: Auditor checks raw output ────────────────────────────────
        auditor = audit_doctor_output(
            raw_doctor_output,
            patient_observed,
            drug_db_path=self.config.drug_db_path,
        )

        auditor_flags = {
            "is_correct": bool(auditor.get("safe", False)),
            "flags": list(auditor.get("flags", [])) if isinstance(auditor.get("flags"), list) else [],
        }

        # ── Step 3: Clinical Verification Layer (2FA safety check) ──────────
        # CVL refines the output using Claude API as senior clinician reviewer
        # This is INDEPENDENT of reward — purely for safety and transparency
        cvl_output: Optional[Dict[str, Any]] = None
        if self.config.enable_cvl:
            try:
                cvl = self._get_cvl()
                if cvl.is_active:
                    cvl_output = cvl.verify(
                        patient_original=patient_original,
                        patient_observed=patient_observed,
                        doctor_output=raw_doctor_output,
                        auditor_flags=auditor_flags,
                    )
            except Exception as e:
                # CVL failure never blocks the episode
                import sys
                print(f"[MedSentinelEnv] CVL error (non-fatal): {e}", file=sys.stderr)

        # ── Step 4: Reward computed on RAW doctor output (not CVL) ──────────
        # This is intentional — we train the doctor agent, not the CVL
        reward, breakdown = compute_reward(
            raw_doctor_output,
            patient_observed,
            auditor_flags=auditor_flags,
            drift_flag=self._current_drift_flag,
            drug_db_path=self.config.drug_db_path,
            icd_db_path=self.config.icd_db_path,
        )

        # ── Step 5: Build info dict ──────────────────────────────────────────
        info: Dict[str, Any] = {
            "patient_id": patient_observed.get("patient_id"),
            "mode": self.config.mode,
            "drift_occurred": self._current_drift_flag,
            "drift_changes": self._current_drift_changes,
            "auditor": auditor,
            "reward_breakdown": breakdown,
            # CVL output included for transparency — None if CVL disabled/unavailable
            "cvl_output": cvl_output,
            "cvl_active": cvl_output is not None,
        }

        done = True
        return float(reward), done, info

