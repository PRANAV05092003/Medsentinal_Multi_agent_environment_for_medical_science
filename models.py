"""
MedSentinel OpenEnv Models
==========================

Pydantic Action / Observation / State models following the openenv-core spec.

These are used by:
- server/medsentinel_environment.py  (environment logic)
- server/app.py                      (FastAPI server via create_app)
- openenv_client.py                  (EnvClient)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import Field

from openenv.core.env_server.types import Action, Observation, State


# ─── Action ──────────────────────────────────────────────────────────────────

class MedSentinelAction(Action):
    """
    Doctor agent output — the action taken in the environment.

    The agent receives a patient record and must return:
    - A clinical diagnosis (ICD-10 code + name)
    - A drug prescription + dosage
    - Whether it detected schema drift
    """

    reasoning: str = Field(
        default="",
        description="Clinical reasoning explaining the diagnosis and treatment choice",
    )
    diagnosis_icd10: str = Field(
        default="",
        description="ICD-10 diagnosis code (e.g. 'I21.9')",
    )
    diagnosis_name: str = Field(
        default="",
        description="Human-readable diagnosis name (e.g. 'STEMI')",
    )
    prescribed_drug: str = Field(
        default="",
        description="Drug to prescribe (e.g. 'nitroglycerin')",
    )
    dosage_mg: Optional[float] = Field(
        default=None,
        description="Dose in milligrams, or null if unknown",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence in the diagnosis, 0.0 to 1.0",
    )
    schema_drift_handled: bool = Field(
        default=False,
        description="True if the agent detected and interpreted renamed schema keys",
    )

    model_config = {
        "extra": "allow",   # allow extra fields from doctor agent output
        "validate_assignment": True,
        "arbitrary_types_allowed": True,
    }


# ─── Observation ─────────────────────────────────────────────────────────────

class MedSentinelObservation(Observation):
    """
    Environment observation returned after reset() and step().

    After reset(): contains the (possibly drifted) patient record.
    After step():  also contains auditor verdict, reward breakdown, and drift info.
    """

    # Patient record (dict because vitals/labs can have any string keys after drift)
    patient_record: Dict[str, Any] = Field(
        default_factory=dict,
        description="Patient record — vitals/labs may have schema-drifted key names",
    )

    # Populated after step()
    auditor_flags: List[str] = Field(
        default_factory=list,
        description="Rule-based auditor flag codes (e.g. ALLERGY_VIOLATION)",
    )
    auditor_safe: bool = Field(
        default=True,
        description="True if the auditor found no safety violations",
    )
    reward_breakdown: Dict[str, Any] = Field(
        default_factory=dict,
        description="Per-component reward breakdown dict",
    )

    # Drift metadata
    drift_occurred: bool = Field(
        default=False,
        description="True if schema drift was applied this episode",
    )
    drift_changes: Dict[str, Dict[str, str]] = Field(
        default_factory=lambda: {"vitals": {}, "lab_results": {}},
        description="Map of original_key -> renamed_key for drifted fields",
    )

    # Ground truth (revealed after step for training transparency)
    ground_truth_diagnosis: str = Field(
        default="",
        description="Correct ICD-10 code (revealed post-step)",
    )

    model_config = {
        "extra": "allow",
        "validate_assignment": True,
        "arbitrary_types_allowed": True,
    }


# ─── State ───────────────────────────────────────────────────────────────────

class MedSentinelState(State):
    """
    Internal environment state — returned by /state endpoint.
    """

    mode: str = Field(default="train", description="'train' or 'test'")
    episode_count: int = Field(default=0, description="Total episodes run so far")
    drift_probability: float = Field(default=0.35, description="Schema drift probability")
    dataset_size: int = Field(default=0, description="Total patient cases in dataset")
    current_patient_id: Optional[str] = Field(
        default=None, description="Patient ID of the current episode"
    )
    last_reward: Optional[float] = Field(
        default=None, description="Reward from the last completed episode"
    )

    model_config = {
        "extra": "allow",
        "validate_assignment": True,
        "arbitrary_types_allowed": True,
    }
