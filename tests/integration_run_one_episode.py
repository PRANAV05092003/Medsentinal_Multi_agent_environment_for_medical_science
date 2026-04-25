#!/usr/bin/env python3
"""
MedSentinel Full Integration Smoke Test (One Episode)
====================================================

This script runs a single end-to-end episode across:
- Environment (`MedSentinelEnv`)
- Schema drift simulation
- Doctor agent (`DoctorAgent`)
- Auditor agent (rule-based)
- Reward function (deterministic)

It is designed to *never crash* in typical dev setups:
- If `data/patient_cases.json` doesn't exist, it uses a small built-in fallback dataset
- If `ANTHROPIC_API_KEY` is missing, it auto-falls back to `DoctorAgent(provider="local")`

Run:
  python tests/integration_run_one_episode.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from typing import Any, Dict, List

# Ensure repo root is importable when running from `tests/`.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from agents.doctor_agent import DoctorAgent  # noqa: E402
from env.medsentinel_env import EnvConfig, MedSentinelEnv  # noqa: E402


DEFAULT_DATASET_PATH = os.path.join("data", "patient_cases.json")


def _fallback_dataset() -> List[Dict[str, Any]]:
    """
    Minimal dataset matching the patient schema required by the env/reward/auditor.
    """
    return [
        {
            "patient_id": "PT-INTEGRATION-1",
            "age": 52,
            "gender": "male",
            "chief_complaint": "crushing chest pain radiating to left arm",
            "vitals": {"heart_rate": 112, "bp_systolic": 96, "bp_diastolic": 62, "respiratory_rate": 20, "spo2": 95, "temperature": 36.9},
            "lab_results": {"troponin_i": 0.38, "sodium": 134, "potassium": 4.2, "creatinine": 1.1, "lactate": 1.8},
            "known_allergies": ["Aspirin"],
            "current_medications": [],
            "ground_truth_diagnosis": "I21.9",
            "safe_drugs": ["Nitroglycerin"],
            "unsafe_drugs": ["Aspirin"],
        }
    ]


def _write_temp_dataset(cases: List[Dict[str, Any]]) -> str:
    """
    Write a temporary dataset file and return its path.
    """
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return path


def _pick_doctor_agent() -> DoctorAgent:
    """
    Prefer Anthropic if API key is available; otherwise use local fallback provider.
    """
    if os.environ.get("ANTHROPIC_API_KEY"):
        return DoctorAgent(provider="anthropic")
    # If .env exists with key, DoctorAgent(provider="anthropic") will work too,
    # but we avoid reading local files here; the agent module will do it if used.
    try:
        return DoctorAgent(provider="anthropic")
    except Exception:
        return DoctorAgent(provider="local")


def main() -> int:
    dataset_path = DEFAULT_DATASET_PATH
    temp_path: str | None = None

    if not os.path.exists(dataset_path):
        temp_path = _write_temp_dataset(_fallback_dataset())
        dataset_path = temp_path

    try:
        env = MedSentinelEnv(
            EnvConfig(
                patient_dataset_path=dataset_path,
                mode="test",
                seed=123,
                drift_probability=1.0,  # force drift so drift info is always present
                max_key_renames_per_section=2,
            )
        )

        patient_obs = env.reset()
        doctor = _pick_doctor_agent()

        doctor_output = doctor.diagnose(patient_obs)
        reward, done, info = env.step(doctor_output)

        # Required prints for judging/debugging.
        print("=== Doctor output ===")
        print("diagnosis_icd10:", doctor_output.get("diagnosis_icd10"))
        print("diagnosis_name :", doctor_output.get("diagnosis_name"))
        print("prescribed_drug:", doctor_output.get("prescribed_drug"))
        print("dosage_mg      :", doctor_output.get("dosage_mg"))
        print("confidence     :", doctor_output.get("confidence"))
        print("schema_drift_handled:", doctor_output.get("schema_drift_handled"))
        print()

        print("=== Reward ===")
        print("reward:", reward)
        print("done  :", done)
        print()

        print("=== Breakdown ===")
        print(json.dumps(info.get("reward_breakdown", {}), ensure_ascii=False, indent=2))
        print()

        print("=== Auditor ===")
        auditor = info.get("auditor", {})
        print("safe :", auditor.get("safe"))
        print("flags:", auditor.get("flags"))
        print("notes:", auditor.get("notes"))
        print()

        print("=== Drift info ===")
        print("drift_occurred:", info.get("drift_occurred"))
        print("drift_changes :", json.dumps(info.get("drift_changes", {}), ensure_ascii=False, indent=2))
        print()

        return 0
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                # Not fatal for a test script.
                pass


if __name__ == "__main__":
    raise SystemExit(main())

