"""
MedSentinel OpenEnv Environment
================================

Proper OpenEnv-compliant environment class inheriting from
openenv.core.env_server.interfaces.Environment.

This wraps the existing MedSentinelEnv gym-style logic inside the
OpenEnv base class API so it works with:
- create_app() for HuggingFace Spaces deployment
- EnvClient for GRPO training integration
- OpenEnv CLI (openenv push, openenv run)

API contract:
  reset(seed, episode_id, **kwargs) -> MedSentinelObservation
  step(action: MedSentinelAction)   -> MedSentinelObservation
  state                             -> MedSentinelState (property)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

# Ensure repo root on path regardless of where this file is imported from
_SERVER_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SERVER_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State as _BaseState

try:
    from ..models import MedSentinelAction, MedSentinelObservation, MedSentinelState
except ImportError:
    from models import MedSentinelAction, MedSentinelObservation, MedSentinelState

from env.medsentinel_env import EnvConfig, MedSentinelEnv
from agents.auditor_agent import audit_doctor_output
from env.reward_system import compute_reward


_DEFAULT_DATASET = str(_REPO_ROOT / "data" / "patient_cases.json")
_DEFAULT_DRUG_DB = str(_REPO_ROOT / "data" / "emergency_drugs.json")
_DEFAULT_ICD_DB = str(_REPO_ROOT / "data" / "icd10_emergency_conditions.json")


class MedSentinelEnvironment(Environment[MedSentinelAction, MedSentinelObservation, MedSentinelState]):
    """
    Multi-agent medical RL environment for MedSentinel.

    Each episode:
      1. reset() picks a patient from the dataset and applies schema drift
      2. The agent calls step() with its diagnosis + drug prescription
      3. The auditor checks safety and the reward function scores the action
      4. Episode ends (done=True) after one step

    Rewards:
      +0.40  correct ICD-10 diagnosis
      +0.20  safe drug (no allergy, not in unsafe list)
      +0.20  correct dosage (within clinical range)
      +0.10  schema drift detected and handled
      +0.10  auditor found no violations
      -0.50  prescribed a drug the patient is allergic to
      -0.30  wrong diagnosis with high confidence (>= 0.8)
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = False

    def __init__(
        self,
        dataset_path: str = _DEFAULT_DATASET,
        drift_probability: float = 0.35,
        max_key_renames_per_section: int = 2,
        mode: str = "train",
        seed: int = 1337,
        test_fraction: float = 0.2,
        drug_db_path: str = _DEFAULT_DRUG_DB,
        icd_db_path: str = _DEFAULT_ICD_DB,
    ) -> None:
        super().__init__()

        self._drug_db_path = drug_db_path
        self._icd_db_path = icd_db_path
        self._drift_probability = drift_probability
        self._seed = seed

        cfg = EnvConfig(
            patient_dataset_path=dataset_path,
            drift_probability=drift_probability,
            max_key_renames_per_section=max_key_renames_per_section,
            mode=mode,
            seed=seed,
            test_fraction=test_fraction,
            drug_db_path=drug_db_path,
            icd_db_path=icd_db_path,
        )
        self._env = MedSentinelEnv(cfg)

        self._episode_id: Optional[str] = None
        self._episode_count: int = 0
        self._last_reward: Optional[float] = None
        self._last_obs: Optional[MedSentinelObservation] = None

    # ─── OpenEnv API ─────────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        mode: Optional[str] = None,
        **kwargs: Any,
    ) -> MedSentinelObservation:
        """
        Start a new episode.

        Returns a MedSentinelObservation containing the (possibly drifted)
        patient record as `patient_record`.
        """
        self._reset_rubric()

        if seed is not None:
            self._env.config.seed = seed
            self._env._rng.seed(seed)

        self._episode_id = episode_id or str(uuid4())
        self._episode_count += 1

        patient_obs = self._env.reset(mode=mode)
        drift_info = self._env.current_drift_info

        obs = MedSentinelObservation(
            patient_record=dict(patient_obs),
            drift_occurred=drift_info["drift_occurred"],
            drift_changes=drift_info["drift_changes"],
            done=False,
            reward=None,
        )
        self._last_obs = obs
        return obs

    def step(
        self,
        action: MedSentinelAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> MedSentinelObservation:
        """
        Evaluate the doctor action and end the episode.

        Accepts MedSentinelAction and returns a MedSentinelObservation
        with reward, auditor result, and breakdown.
        """
        # Convert Pydantic action to plain dict for existing env logic
        if isinstance(action, MedSentinelAction):
            doctor_output: Dict[str, Any] = action.model_dump(exclude={"metadata"})
        elif isinstance(action, dict):
            doctor_output = action
        else:
            doctor_output = {}

        reward_float, done, info = self._env.step(doctor_output)
        self._last_reward = reward_float

        auditor = info.get("auditor", {})
        breakdown = info.get("reward_breakdown", {})
        drift_info = self._env.current_drift_info

        # Get ground truth from original patient record
        original = self._env.current_patient_original or {}
        ground_truth = str(original.get("ground_truth_diagnosis", ""))

        obs = MedSentinelObservation(
            patient_record=dict(self._env.current_patient_observed or {}),
            auditor_flags=list(auditor.get("flags", [])),
            auditor_safe=bool(auditor.get("safe", False)),
            reward_breakdown=dict(breakdown),
            drift_occurred=drift_info["drift_occurred"],
            drift_changes=drift_info["drift_changes"],
            ground_truth_diagnosis=ground_truth,
            done=True,
            reward=reward_float,
        )
        self._last_obs = obs
        return obs

    @property
    def state(self) -> MedSentinelState:
        """Return current internal state."""
        original = self._env.current_patient_original or {}
        return MedSentinelState(
            episode_id=self._episode_id,
            step_count=self._episode_count,
            mode=self._env.config.mode,
            episode_count=self._episode_count,
            drift_probability=self._drift_probability,
            dataset_size=len(self._env._dataset),
            current_patient_id=str(original.get("patient_id", "")) or None,
            last_reward=self._last_reward,
        )

    def get_metadata(self):
        from openenv.core.env_server.interfaces import EnvironmentMetadata
        return EnvironmentMetadata(
            name="MedSentinel",
            description=(
                "Multi-agent RL environment for emergency clinical decision-making "
                "with adversarial schema drift. Doctor agent learns to diagnose patients "
                "and prescribe safe treatments under key-rename attacks."
            ),
            version="1.0.0",
        )

    def close(self) -> None:
        pass
