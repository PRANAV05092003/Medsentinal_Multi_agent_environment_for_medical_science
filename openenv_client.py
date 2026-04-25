"""
MedSentinel OpenEnv Client
===========================

Proper EnvClient subclass for the MedSentinel environment.
Uses the openenv-core WebSocket-based client with sync wrapper.

Usage (sync, e.g. in training scripts):
    from openenv_client import MedSentinelEnv, make_client
    from models import MedSentinelAction

    env = make_client()
    with env:
        result = env.reset(seed=42)
        patient = result.observation.patient_record

        action = MedSentinelAction(
            reasoning="Elevated troponin and chest pain suggest STEMI.",
            diagnosis_icd10="I21.9",
            diagnosis_name="STEMI",
            prescribed_drug="nitroglycerin",
            dosage_mg=0.4,
            confidence=0.9,
            schema_drift_handled=False,
        )
        result = env.step(action)
        print(f"Reward: {result.reward}")

HuggingFace Spaces:
    env = make_client(base_url="https://<your-space>.hf.space")
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from models import MedSentinelAction, MedSentinelObservation, MedSentinelState


class MedSentinelEnv(
    EnvClient[MedSentinelAction, MedSentinelObservation, MedSentinelState]
):
    """
    Client for the MedSentinel OpenEnv environment.

    Maintains a persistent WebSocket session with the MedSentinel server.
    Compatible with local servers and HuggingFace Spaces.
    """

    def _step_payload(self, action: MedSentinelAction) -> Dict[str, Any]:
        """Serialize MedSentinelAction to JSON payload."""
        return action.model_dump(exclude={"metadata"})

    def _parse_result(
        self, payload: Dict[str, Any]
    ) -> StepResult[MedSentinelObservation]:
        """Parse server step response into StepResult."""
        obs_data = payload.get("observation", {})
        observation = MedSentinelObservation(
            patient_record=obs_data.get("patient_record", {}),
            auditor_flags=obs_data.get("auditor_flags", []),
            auditor_safe=obs_data.get("auditor_safe", True),
            reward_breakdown=obs_data.get("reward_breakdown", {}),
            drift_occurred=obs_data.get("drift_occurred", False),
            drift_changes=obs_data.get("drift_changes", {"vitals": {}, "lab_results": {}}),
            ground_truth_diagnosis=obs_data.get("ground_truth_diagnosis", ""),
            done=payload.get("done", True),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", True),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> MedSentinelState:
        """Parse /state response into MedSentinelState."""
        return MedSentinelState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            mode=payload.get("mode", "train"),
            episode_count=payload.get("episode_count", 0),
            drift_probability=payload.get("drift_probability", 0.35),
            dataset_size=payload.get("dataset_size", 0),
            current_patient_id=payload.get("current_patient_id"),
            last_reward=payload.get("last_reward"),
        )


def make_client(
    base_url: Optional[str] = None,
    timeout_s: float = 60.0,
) -> Any:
    """
    Create a sync MedSentinel client.

    Example:
        env = make_client()
        with env:
            result = env.reset()
    """
    url = (
        base_url
        or os.environ.get("MEDSENTINEL_SERVER_URL")
        or "http://localhost:7860"
    )
    return MedSentinelEnv(base_url=url, message_timeout_s=timeout_s).sync()
