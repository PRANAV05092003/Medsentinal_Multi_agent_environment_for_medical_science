#!/usr/bin/env python3
"""
MedSentinel Clinical Verification Layer (CVL)
==============================================

This is the 2-Factor Authentication for medical decisions.

The problem it solves:
  In real medicine, even a 0.1% error can kill someone. The doctor agent
  (our RL-trained model) makes its best guess — but it can hallucinate,
  miss drug interactions, or output a plausible-sounding but wrong diagnosis.

  The CVL acts as a senior clinician reviewer sitting between the doctor
  agent's raw output and the final answer delivered to the environment.

How it works:
  1. Doctor agent proposes: diagnosis + drug + dose
  2. CVL receives:
     - The original patient record (before schema drift)
     - The drifted patient record (what the doctor actually saw)
     - The doctor's proposed output
     - The auditor's safety flags
  3. CVL sends all of this to Claude API as a structured verification prompt
  4. Claude acts as a "senior clinician" — it:
     - Confirms or overrides the diagnosis
     - Validates drug appropriateness
     - Checks dose against patient weight/condition
     - Flags any dangerous combinations the auditor missed
     - Refines the reasoning
  5. Returns a verified, refined output — always

Key design decisions:
  - CVL is NOT part of the reward system. The reward only uses the
    doctor agent's raw output. CVL is a safety layer on top.
  - CVL ALWAYS returns something — even if API fails, it falls back
    gracefully to the doctor's original output with a warning flag.
  - CVL output has an extra key: `cvl_verified` (bool) and
    `cvl_changes` (list of what was changed).
  - If Claude API is not configured, CVL runs in "pass-through" mode
    and returns the doctor output unchanged.

Usage:
  from agents.clinical_verification_layer import ClinicalVerificationLayer

  cvl = ClinicalVerificationLayer()
  verified = cvl.verify(
      patient_original=patient_dict,
      patient_observed=drifted_patient_dict,
      doctor_output=doctor_agent_output,
      auditor_flags=auditor_result,
  )
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Mapping, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def _eprint(msg: str) -> None:
    print(msg, file=sys.stderr)


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _get_api_key() -> Optional[str]:
    """Load Anthropic API key from .env or environment. Returns None if not found."""
    env_path = os.path.join(_repo_root(), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("ANTHROPIC_API_KEY="):
                    val = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if val:
                        return val
    return os.environ.get("ANTHROPIC_API_KEY") or None


def _http_post(url: str, headers: Dict, payload: Dict, timeout: int = 45) -> Dict:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = Request(url, data=body, method="POST")
    for k, v in headers.items():
        req.add_header(k, v)
    with urlopen(req, timeout=timeout) as resp:
        charset = resp.headers.get_content_charset() or "utf-8"
        return json.loads(resp.read().decode(charset, errors="replace"))


# ─── Prompt builders ─────────────────────────────────────────────────────────

_CVL_SYSTEM_PROMPT = """You are the Clinical Verification Layer (CVL) — a senior emergency physician 
reviewing another doctor's diagnostic and treatment decision before it reaches the patient.

Your job is NOT to judge. Your job is to verify, refine, and ensure safety.

You will receive:
1. The patient's original record (complete clinical picture)
2. What the doctor agent actually saw (may have renamed/drifted field names)
3. The doctor agent's proposed diagnosis and treatment
4. Any safety flags from the automated auditor

Your task:
- Verify the diagnosis makes clinical sense given the patient data
- Confirm the drug is appropriate for this specific diagnosis AND this specific patient
- Validate the dosage is within safe clinical range for this patient
- Check for any drug interactions with current medications not caught by the auditor
- Check for allergy conflicts
- If the diagnosis/drug/dose is correct: confirm and refine the reasoning
- If something is wrong: override it with the correct answer and explain why

Return ONLY valid JSON with exactly these keys:
{
  "verified_diagnosis_icd10": "<ICD-10 code>",
  "verified_diagnosis_name": "<condition name>",
  "verified_drug": "<drug name>",
  "verified_dosage_mg": <number or null>,
  "verified_confidence": <0.0 to 1.0>,
  "verified_reasoning": "<comprehensive clinical reasoning, minimum 80 words>",
  "schema_drift_handled": <true or false>,
  "cvl_verified": true,
  "cvl_changes": ["list of what you changed vs doctor output, empty list if nothing changed"],
  "cvl_risk_flags": ["any remaining risk flags even after correction, empty if none"],
  "cvl_notes": "<brief note about the verification decision>"
}

CRITICAL RULES:
- Return ONLY the JSON object. No markdown, no prose before or after.
- verified_drug MUST NOT match any allergy in the patient record
- verified_drug MUST be appropriate for the verified diagnosis
- If the doctor was right: cvl_changes = []
- If you changed something: explain exactly what and why in cvl_changes
- Be a safety net, not a second-guesser. Override only when clearly necessary."""


def _build_verification_prompt(
    patient_original: Mapping[str, Any],
    patient_observed: Mapping[str, Any],
    doctor_output: Mapping[str, Any],
    auditor_flags: Dict[str, Any],
) -> str:
    """Build the full verification prompt with all context."""

    # Hide ground truth from CVL — it should reason from symptoms, not cheat
    patient_clean = {
        k: v for k, v in patient_original.items()
        if k not in ("ground_truth_diagnosis",)
    }
    patient_observed_clean = {
        k: v for k, v in patient_observed.items()
        if k not in ("ground_truth_diagnosis",)
    }

    sections = [
        "=== ORIGINAL PATIENT RECORD (complete) ===",
        json.dumps(patient_clean, indent=2),
        "",
        "=== WHAT THE DOCTOR AGENT SAW (may have schema drift) ===",
        json.dumps(patient_observed_clean, indent=2),
        "",
        "=== DOCTOR AGENT'S PROPOSED OUTPUT ===",
        json.dumps(dict(doctor_output), indent=2),
        "",
        "=== AUTOMATED AUDITOR FLAGS ===",
        json.dumps(auditor_flags, indent=2),
        "",
        "Please verify this clinical decision and return the verified output as JSON.",
    ]
    return "\n".join(sections)


# ─── CVL output parser ────────────────────────────────────────────────────────

def _parse_cvl_response(text: str) -> Dict[str, Any]:
    """Extract and validate CVL JSON from Claude response."""
    text = text.strip()

    # Direct parse
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Find JSON block
    start = text.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start:i + 1])
                        if isinstance(obj, dict):
                            return obj
                    except Exception:
                        pass
                    break

    return {}


def _normalize_cvl_output(
    cvl_raw: Dict[str, Any],
    doctor_output: Mapping[str, Any],
    fallback_reason: str = "",
) -> Dict[str, Any]:
    """
    Normalize CVL output. If CVL fields are missing, fall back to doctor output.
    Always returns a complete, safe dict.
    """
    def _str(val: Any, fallback: str = "") -> str:
        return str(val).strip() if isinstance(val, str) and str(val).strip() else fallback

    def _float01(val: Any, fallback: float = 0.0) -> float:
        try:
            v = float(val)
            return max(0.0, min(1.0, v))
        except Exception:
            return fallback

    def _list(val: Any) -> List[str]:
        if isinstance(val, list):
            return [str(x) for x in val]
        return []

    # Fall back to doctor output for any missing clinical fields
    return {
        # Verified clinical decision
        "reasoning": _str(
            cvl_raw.get("verified_reasoning"),
            fallback=_str(doctor_output.get("reasoning"), "No reasoning provided.")
        ),
        "diagnosis_icd10": _str(
            cvl_raw.get("verified_diagnosis_icd10"),
            fallback=_str(doctor_output.get("diagnosis_icd10"), "")
        ),
        "diagnosis_name": _str(
            cvl_raw.get("verified_diagnosis_name"),
            fallback=_str(doctor_output.get("diagnosis_name"), "")
        ),
        "prescribed_drug": _str(
            cvl_raw.get("verified_drug"),
            fallback=_str(doctor_output.get("prescribed_drug"), "")
        ),
        "dosage_mg": (
            float(cvl_raw["verified_dosage_mg"])
            if cvl_raw.get("verified_dosage_mg") is not None
            and isinstance(cvl_raw["verified_dosage_mg"], (int, float))
            else doctor_output.get("dosage_mg")
        ),
        "confidence": _float01(
            cvl_raw.get("verified_confidence"),
            fallback=_float01(doctor_output.get("confidence"), 0.5)
        ),
        "schema_drift_handled": bool(
            cvl_raw.get("schema_drift_handled",
                        doctor_output.get("schema_drift_handled", False))
        ),

        # CVL metadata — not used in reward, only for transparency
        "cvl_verified": bool(cvl_raw.get("cvl_verified", False)),
        "cvl_changes": _list(cvl_raw.get("cvl_changes", [])),
        "cvl_risk_flags": _list(cvl_raw.get("cvl_risk_flags", [])),
        "cvl_notes": _str(cvl_raw.get("cvl_notes", fallback_reason)),
        "cvl_fallback": bool(fallback_reason),
    }


# ─── Main CVL class ───────────────────────────────────────────────────────────

class ClinicalVerificationLayer:
    """
    The 2-Factor Authentication layer for MedSentinel clinical decisions.

    Sits between the doctor agent and the environment's reward/auditor system.
    Uses Claude API as a senior clinician reviewer.

    NOT connected to the reward system — purely a safety/transparency layer.
    """

    def __init__(
        self,
        *,
        model: str = "claude-3-5-sonnet-20241022",
        max_tokens: int = 1500,
        temperature: float = 0.1,   # Low temp for conservative verification
        timeout_s: int = 45,
        enabled: bool = True,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout_s = timeout_s
        self.enabled = enabled

        self._api_key = _get_api_key()
        if not self._api_key:
            _eprint("[CVL] No Anthropic API key found — running in pass-through mode.")
            self.enabled = False

    @property
    def is_active(self) -> bool:
        """True if CVL is actually calling Claude API."""
        return self.enabled and bool(self._api_key)

    def verify(
        self,
        *,
        patient_original: Mapping[str, Any],
        patient_observed: Mapping[str, Any],
        doctor_output: Mapping[str, Any],
        auditor_flags: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Verify a doctor agent's clinical decision.

        Args:
            patient_original: The patient record before any schema drift.
            patient_observed: The patient record the doctor actually saw (may have drift).
            doctor_output:    The doctor agent's proposed diagnosis + treatment.
            auditor_flags:    Output from the rule-based auditor agent (optional).

        Returns:
            A verified, refined clinical decision dict.
            Always returns something — never raises.
            Includes CVL metadata keys: cvl_verified, cvl_changes, cvl_risk_flags,
            cvl_notes, cvl_fallback.
        """
        if auditor_flags is None:
            auditor_flags = {}

        # Pass-through mode when API not configured
        if not self.is_active:
            result = dict(doctor_output)
            result.update({
                "cvl_verified": False,
                "cvl_changes": [],
                "cvl_risk_flags": [],
                "cvl_notes": "CVL in pass-through mode — no API key configured.",
                "cvl_fallback": True,
            })
            return result

        try:
            return self._call_claude(
                patient_original=patient_original,
                patient_observed=patient_observed,
                doctor_output=doctor_output,
                auditor_flags=auditor_flags,
            )
        except Exception as e:
            _eprint(f"[CVL] Verification failed ({e}), returning doctor output with fallback flag.")
            return _normalize_cvl_output(
                cvl_raw={},
                doctor_output=doctor_output,
                fallback_reason=f"CVL API call failed: {e}",
            )

    def _call_claude(
        self,
        *,
        patient_original: Mapping[str, Any],
        patient_observed: Mapping[str, Any],
        doctor_output: Mapping[str, Any],
        auditor_flags: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Make the actual Claude API call for verification."""

        user_content = _build_verification_prompt(
            patient_original=patient_original,
            patient_observed=patient_observed,
            doctor_output=doctor_output,
            auditor_flags=auditor_flags,
        )

        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "system": _CVL_SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": user_content}],
        }

        headers = {
            "content-type": "application/json",
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01",
        }

        resp = _http_post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            payload=payload,
            timeout=self.timeout_s,
        )

        # Extract text from response
        content = resp.get("content", [])
        texts = [
            block["text"] for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        raw_text = "\n".join(texts).strip()

        if not raw_text:
            raise ValueError("Empty response from Claude API")

        cvl_raw = _parse_cvl_response(raw_text)
        if not cvl_raw:
            raise ValueError(f"Could not parse CVL JSON from response: {raw_text[:300]}")

        result = _normalize_cvl_output(
            cvl_raw=cvl_raw,
            doctor_output=doctor_output,
            fallback_reason="",
        )

        # Log what changed for transparency
        changes = result.get("cvl_changes", [])
        if changes:
            _eprint(f"[CVL] Changes made: {changes}")
        else:
            _eprint("[CVL] Doctor output verified — no changes needed.")

        return result


# ─── Convenience function ─────────────────────────────────────────────────────

_default_cvl: Optional[ClinicalVerificationLayer] = None


def get_default_cvl() -> ClinicalVerificationLayer:
    """Get or create the default singleton CVL instance."""
    global _default_cvl
    if _default_cvl is None:
        _default_cvl = ClinicalVerificationLayer()
    return _default_cvl


def verify_clinical_decision(
    patient_original: Mapping[str, Any],
    patient_observed: Mapping[str, Any],
    doctor_output: Mapping[str, Any],
    auditor_flags: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Convenience wrapper. Verify a clinical decision using the default CVL.

    This is the main entry point for the 2FA verification pipeline.
    """
    return get_default_cvl().verify(
        patient_original=patient_original,
        patient_observed=patient_observed,
        doctor_output=doctor_output,
        auditor_flags=auditor_flags,
    )
