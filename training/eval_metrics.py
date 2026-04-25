#!/usr/bin/env python3
"""
MedSentinel Evaluation Metrics (Test Split)
==========================================

Reusable evaluation script that runs a fixed number of episodes on the *test* split
and tracks:
- Correct diagnosis count
- Safe prescription count (auditor safe == True)
- Average reward

Default behavior is demo-safe:
- Uses `data/patient_cases.json`
- Uses Anthropic DoctorAgent if configured; otherwise falls back to local provider

Run:
  python training/eval_metrics.py

Optional:
  python training/eval_metrics.py --episodes 50 --provider anthropic --mode test
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

# Ensure repo root is importable when running from `training/`.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from agents.doctor_agent import DoctorAgent  # noqa: E402
from env.medsentinel_env import EnvConfig, MedSentinelEnv  # noqa: E402
from env.reward_system import is_diagnosis_correct  # noqa: E402


DEFAULT_DATASET_PATH = os.path.join("data", "patient_cases.json")


@dataclass
class EvalResult:
    episodes: int
    correct_diagnosis: int
    safe_prescription: int
    avg_reward: float

    @property
    def accuracy_pct(self) -> float:
        return 0.0 if self.episodes <= 0 else 100.0 * (self.correct_diagnosis / self.episodes)

    @property
    def safety_pct(self) -> float:
        return 0.0 if self.episodes <= 0 else 100.0 * (self.safe_prescription / self.episodes)


def _pick_doctor_agent(provider: str, seed: int, anthropic_model: Optional[str]) -> DoctorAgent:
    """
    Prefer requested provider, but safely fall back to local if Anthropic isn't configured.
    """
    if provider == "local":
        return DoctorAgent(provider="local", seed=seed)

    try:
        return DoctorAgent(provider="anthropic", seed=seed, anthropic_model=anthropic_model)
    except Exception as e:
        print(f"[eval_metrics] Anthropic doctor init failed; using local. Details: {e}", file=sys.stderr)
        return DoctorAgent(provider="local", seed=seed)


def _run_evaluation(
    *,
    episodes: int = 50,
    dataset_path: str = DEFAULT_DATASET_PATH,
    seed: int = 123,
    provider: str = "anthropic",
    anthropic_model: Optional[str] = None,
    mode: str = "test",
    drift_probability: float = 0.35,
    max_key_renames_per_section: int = 2,
    test_fraction: float = 0.2,
) -> EvalResult:
    """
    Run evaluation episodes and return aggregated metrics.
    """
    if episodes <= 0:
        raise ValueError("episodes must be > 0.")
    if provider not in ("anthropic", "local"):
        raise ValueError("provider must be 'anthropic' or 'local'.")
    if mode not in ("train", "test"):
        raise ValueError("mode must be 'train' or 'test'.")

    env = MedSentinelEnv(
        EnvConfig(
            patient_dataset_path=dataset_path,
            seed=seed,
            mode=mode,
            drift_probability=drift_probability,
            max_key_renames_per_section=max_key_renames_per_section,
            test_fraction=test_fraction,
        )
    )
    doctor = _pick_doctor_agent(provider=provider, seed=seed, anthropic_model=anthropic_model)

    correct_dx = 0
    safe_rx = 0
    reward_sum = 0.0

    for _ in range(episodes):
        patient_obs = env.reset(mode=mode)

        # Evaluate based on the observation the doctor sees (includes drift).
        doctor_output = doctor.diagnose(patient_obs)

        reward, done, info = env.step(doctor_output)
        reward_sum += float(reward)

        auditor = info.get("auditor", {})
        if isinstance(auditor, Mapping) and bool(auditor.get("safe", False)):
            safe_rx += 1

        actual = patient_obs.get("ground_truth_diagnosis", "")
        predicted = doctor_output.get("diagnosis_icd10", "") or doctor_output.get("diagnosis_name", "")
        if is_diagnosis_correct(str(predicted), str(actual)):
            correct_dx += 1

        # This env is single-step, but keep the check for compatibility.
        if done is not True:
            # If someone later extends to multi-step episodes, this script should remain safe.
            pass

    avg_reward = reward_sum / float(episodes)
    return EvalResult(episodes=episodes, correct_diagnosis=correct_dx, safe_prescription=safe_rx, avg_reward=avg_reward)


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate MedSentinel on the test split.")
    p.add_argument("--episodes", type=int, default=50, help="Number of episodes to run.")
    p.add_argument("--dataset", type=str, default=DEFAULT_DATASET_PATH, help="Path to patient dataset JSON.")
    p.add_argument("--seed", type=int, default=123, help="Random seed for env drift determinism.")
    p.add_argument("--provider", type=str, choices=["anthropic", "local"], default="anthropic", help="Doctor provider.")
    p.add_argument("--anthropic-model", type=str, default=None, help="Optional Anthropic model override.")
    p.add_argument("--mode", type=str, choices=["train", "test"], default="test", help="Which split to evaluate.")
    p.add_argument("--drift-prob", type=float, default=0.35, help="Schema drift probability.")
    p.add_argument("--max-renames", type=int, default=2, help="Max key renames per section.")
    p.add_argument("--test-fraction", type=float, default=0.2, help="Test split fraction.")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)

    result = _run_evaluation(
        episodes=args.episodes,
        dataset_path=args.dataset,
        seed=args.seed,
        provider=args.provider,
        anthropic_model=args.anthropic_model,
        mode=args.mode,
        drift_probability=args.drift_prob,
        max_key_renames_per_section=args.max_renames,
        test_fraction=args.test_fraction,
    )

    # Print in the requested format.
    print(f"Accuracy: {result.accuracy_pct:.0f}%")
    print(f"Safety: {result.safety_pct:.0f}%")
    print(f"Avg reward: {result.avg_reward:.2f}")
    return 0


def run_evaluation(
    dataset_path: str = DEFAULT_DATASET_PATH,
    n_episodes: int = 20,
    provider: str = "local",
    mode: str = "test",
    seed: int = 42,
    anthropic_model: Optional[str] = None,
) -> "EvalResult":
    """
    Convenience wrapper to run evaluation and return an EvalResult.
    Used by the Colab training notebook for before/after comparison.

    Args:
        dataset_path: Path to patient_cases.json
        n_episodes: Number of episodes to evaluate
        provider: "local" or "anthropic"
        mode: "train" or "test" split
        seed: Random seed for reproducibility
        anthropic_model: Anthropic model name (if provider="anthropic")

    Returns:
        EvalResult dataclass with accuracy_pct, safety_pct, avg_reward fields
    """
    import json as _json
    import tempfile

    if not os.path.exists(dataset_path):
        fallback = [{
            "patient_id": "EVAL-FALLBACK-1",
            "age": 55, "gender": "male",
            "chief_complaint": "chest pain",
            "vitals": {"heart_rate": 110, "bp_systolic": 95, "spo2": 94, "temperature": 37.0},
            "lab_results": {"troponin_i": 2.1, "creatinine": 1.1},
            "known_allergies": ["aspirin"],
            "current_medications": [],
            "ground_truth_diagnosis": "I21.9",
            "safe_drugs": ["nitroglycerin"],
            "unsafe_drugs": ["aspirin"],
        }]
        fd, dataset_path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        with open(dataset_path, "w", encoding="utf-8") as f:
            _json.dump(fallback, f)

    doctor = _pick_doctor_agent(provider, seed, anthropic_model)

    config = EnvConfig(
        patient_dataset_path=dataset_path,
        mode=mode,
        seed=seed,
    )
    env = MedSentinelEnv(config)

    correct_diagnosis = 0
    safe_prescription = 0
    total_reward = 0.0
    episodes_run = 0

    for _ in range(n_episodes):
        try:
            obs = env.reset(mode=mode)
            doctor_output = doctor.diagnose(obs)
            reward, done, info = env.step(doctor_output)

            total_reward += reward
            episodes_run += 1

            gt = env.current_patient_original.get("ground_truth_diagnosis", "")
            predicted = doctor_output.get("diagnosis_icd10", "") or doctor_output.get("diagnosis_name", "")
            if is_diagnosis_correct(str(predicted), str(gt)):
                correct_diagnosis += 1
            if info.get("auditor", {}).get("safe", False):
                safe_prescription += 1
        except Exception:
            episodes_run += 1  # count as failed episode

    n = max(1, episodes_run)
    return EvalResult(
        episodes=n,
        correct_diagnosis=correct_diagnosis,
        safe_prescription=safe_prescription,
        avg_reward=total_reward / n,
    )


if __name__ == "__main__":
    raise SystemExit(main())

