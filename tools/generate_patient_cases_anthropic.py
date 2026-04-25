#!/usr/bin/env python3
"""
MedSentinel - Synthetic patient case generator (Anthropic API)

This script generates synthetic emergency patient cases in batches and saves them as JSON.

Key requirements satisfied:
- Loads Anthropic API key from `.env` (no external dotenv dependency)
- Generates 100 patient cases in batches
- Enforces valid JSON output via strict parsing + retry logic
- Validates each case before saving
- Saves to `data/patient_cases.json`

Usage (PowerShell):
  python tools/generate_patient_cases_anthropic.py

Optional:
  python tools/generate_patient_cases_anthropic.py --count 100 --batch-size 10
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


REQUIRED_CASE_KEYS = (
    "patient_id",
    "age",
    "gender",
    "chief_complaint",
    "vitals",
    "lab_results",
    "known_allergies",
    "current_medications",
    "ground_truth_diagnosis",
    "safe_drugs",
    "unsafe_drugs",
)


class CaseValidationError(ValueError):
    """Raised when a generated case fails schema/type validation."""


def _eprint(msg: str) -> None:
    print(msg, file=sys.stderr)


def load_dotenv(dotenv_path: str) -> Dict[str, str]:
    """
    Minimal `.env` loader.
    - Supports KEY=VALUE lines
    - Ignores blank lines and comments starting with '#'
    - Supports quoted values with single/double quotes
    """
    env: Dict[str, str] = {}
    if not os.path.exists(dotenv_path):
        return env

    with open(dotenv_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if not key:
                continue
            if (len(value) >= 2) and ((value[0] == value[-1]) and value[0] in ("'", '"')):
                value = value[1:-1]
            env[key] = value
    return env


def get_anthropic_api_key(dotenv_path: str) -> str:
    """
    Loads ANTHROPIC_API_KEY from `.env` (preferred) or process env.
    """
    file_env = load_dotenv(dotenv_path)
    api_key = file_env.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing ANTHROPIC_API_KEY. Add it to a `.env` file at the repo root "
            "or export it as an environment variable."
        )
    return api_key


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)


def _extract_first_json_block(text: str) -> str:
    """
    Extract the first JSON object/array substring from model output.
    This makes the script resilient to occasional leading/trailing prose.
    """
    # Fast path: if it's already valid JSON, return as-is.
    t = text.strip()
    if t.startswith("{") or t.startswith("["):
        return t

    # Otherwise, try to find the first { ... } or [ ... ] block.
    # We use a simple heuristic (balanced braces/brackets scanning) rather than regex.
    for start_char, end_char in (("{", "}"), ("[", "]")):
        start = t.find(start_char)
        if start == -1:
            continue
        depth = 0
        for i in range(start, len(t)):
            ch = t[i]
            if ch == start_char:
                depth += 1
            elif ch == end_char:
                depth -= 1
                if depth == 0:
                    return t[start : i + 1].strip()
    return t


def parse_json_strict(text: str) -> Any:
    """
    Strict JSON parsing with helpful errors.
    Attempts to extract a JSON block if the model output includes extra text.
    """
    candidate = _extract_first_json_block(text)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as e:
        preview = candidate[:500].replace("\n", "\\n")
        raise json.JSONDecodeError(
            f"{e.msg}. Candidate preview: {preview}",
            e.doc,
            e.pos,
        ) from e


def is_nonempty_str(x: Any) -> bool:
    return isinstance(x, str) and bool(x.strip())


def _as_list_of_str(value: Any, field: str) -> List[str]:
    if not isinstance(value, list):
        raise CaseValidationError(f"Field `{field}` must be a list.")
    out: List[str] = []
    for i, item in enumerate(value):
        if not is_nonempty_str(item):
            raise CaseValidationError(f"Field `{field}[{i}]` must be a non-empty string.")
        out.append(item.strip())
    return out


def _as_int_in_range(value: Any, field: str, min_v: int, max_v: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CaseValidationError(f"Field `{field}` must be an integer.")
    if not (min_v <= value <= max_v):
        raise CaseValidationError(f"Field `{field}` must be in range [{min_v}, {max_v}].")
    return value


def validate_case(case: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate a single patient case and return a normalized copy.
    Raises CaseValidationError on invalid data.
    """
    if not isinstance(case, dict):
        raise CaseValidationError("Case must be a JSON object.")

    missing = [k for k in REQUIRED_CASE_KEYS if k not in case]
    if missing:
        raise CaseValidationError(f"Missing required keys: {missing}")

    normalized: Dict[str, Any] = dict(case)

    if not is_nonempty_str(normalized["patient_id"]):
        raise CaseValidationError("`patient_id` must be a non-empty string.")

    normalized["age"] = _as_int_in_range(normalized["age"], "age", 0, 120)

    if normalized["gender"] not in ("male", "female", "other", "unknown"):
        raise CaseValidationError("`gender` must be one of: male, female, other, unknown.")

    if not is_nonempty_str(normalized["chief_complaint"]):
        raise CaseValidationError("`chief_complaint` must be a non-empty string.")

    if not isinstance(normalized["vitals"], dict) or not normalized["vitals"]:
        raise CaseValidationError("`vitals` must be a non-empty object.")
    if not isinstance(normalized["lab_results"], dict) or not normalized["lab_results"]:
        raise CaseValidationError("`lab_results` must be a non-empty object.")

    normalized["known_allergies"] = _as_list_of_str(normalized["known_allergies"], "known_allergies")
    normalized["current_medications"] = _as_list_of_str(normalized["current_medications"], "current_medications")

    if not is_nonempty_str(normalized["ground_truth_diagnosis"]):
        raise CaseValidationError("`ground_truth_diagnosis` must be a non-empty string.")

    normalized["safe_drugs"] = _as_list_of_str(normalized["safe_drugs"], "safe_drugs")
    normalized["unsafe_drugs"] = _as_list_of_str(normalized["unsafe_drugs"], "unsafe_drugs")

    # Safety: ensure lists don't overlap in an obvious way.
    safe_set = {d.lower() for d in normalized["safe_drugs"]}
    unsafe_set = {d.lower() for d in normalized["unsafe_drugs"]}
    overlap = sorted(safe_set.intersection(unsafe_set))
    if overlap:
        raise CaseValidationError(f"`safe_drugs` and `unsafe_drugs` overlap: {overlap}")

    return normalized


def validate_cases(cases: Any) -> List[Dict[str, Any]]:
    if not isinstance(cases, list):
        raise CaseValidationError("Top-level output must be a JSON array of cases.")
    validated: List[Dict[str, Any]] = []
    for idx, item in enumerate(cases):
        try:
            validated.append(validate_case(item))
        except CaseValidationError as e:
            raise CaseValidationError(f"Case[{idx}] invalid: {e}") from e
    return validated


@dataclass(frozen=True)
class AnthropicConfig:
    api_key: str
    model: str
    max_tokens: int
    temperature: float
    anthropic_version: str = "2023-06-01"
    endpoint: str = "https://api.anthropic.com/v1/messages"
    request_timeout_s: int = 60


def _http_post_json(url: str, headers: Dict[str, str], payload: Dict[str, Any], timeout_s: int) -> Dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = Request(url, data=body, method="POST")
    for k, v in headers.items():
        req.add_header(k, v)
    with urlopen(req, timeout=timeout_s) as resp:
        charset = resp.headers.get_content_charset() or "utf-8"
        raw = resp.read().decode(charset, errors="replace")
        return json.loads(raw)


def call_anthropic_generate_cases(
    cfg: AnthropicConfig,
    batch_size: int,
    batch_index: int,
    seed: int,
    previous_error: Optional[str] = None,
) -> str:
    """
    Calls Anthropic Messages API and returns the model text content.
    We request the model to output ONLY a JSON array of `batch_size` cases.
    """
    # Keep prompt deterministic-ish by including seed/batch metadata.
    # This also helps the auditor agent later if you want reproducibility.
    system_parts = [
        "You generate synthetic emergency department patient cases for RL simulation.",
        "Return ONLY valid JSON, with no markdown, no code fences, and no extra text.",
        "Output MUST be a JSON array of objects. Do not wrap in a top-level object.",
        "All keys must be present and spelled exactly as required.",
        "Use realistic, medically plausible values. Keep units implicit but consistent.",
        "Use gender values strictly from: male, female, other, unknown.",
        "Ensure safe_drugs and unsafe_drugs do not overlap.",
    ]
    if previous_error:
        # Feeding the parsing/validation error back to the model typically improves compliance.
        system_parts.append(f"Previous output failed validation: {previous_error}")

    required_schema = {
        "patient_id": "string, unique within the batch (format: PT-<uuid4> is okay)",
        "age": "integer 0-120",
        "gender": "male|female|other|unknown",
        "chief_complaint": "short string",
        "vitals": "object with keys like hr, bp_systolic, bp_diastolic, rr, temp_c, spo2_pct",
        "lab_results": "object with keys like wbc, hb, platelets, sodium, potassium, creatinine, lactate, troponin",
        "known_allergies": "list of strings (may be empty list)",
        "current_medications": "list of strings (may be empty list)",
        "ground_truth_diagnosis": "string (single most likely ED diagnosis)",
        "safe_drugs": "list of strings (ED-appropriate, consistent with allergies/meds)",
        "unsafe_drugs": "list of strings (contraindicated or risky for this patient)",
    }

    user_prompt = (
        f"Generate exactly {batch_size} distinct patient cases.\n"
        f"Batch index: {batch_index}\n"
        f"Seed: {seed}\n"
        "Return a JSON array ONLY.\n"
        "Schema (all required):\n"
        f"{json.dumps(required_schema, ensure_ascii=False)}\n"
    )

    payload: Dict[str, Any] = {
        "model": cfg.model,
        "max_tokens": cfg.max_tokens,
        "temperature": cfg.temperature,
        "system": "\n".join(system_parts),
        "messages": [{"role": "user", "content": user_prompt}],
    }

    headers = {
        "content-type": "application/json",
        "x-api-key": cfg.api_key,
        "anthropic-version": cfg.anthropic_version,
    }

    try:
        resp = _http_post_json(cfg.endpoint, headers=headers, payload=payload, timeout_s=cfg.request_timeout_s)
    except HTTPError as e:
        # Provide useful context on auth/limits/errors.
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""
        raise RuntimeError(f"Anthropic HTTPError {e.code}: {body[:1000]}") from e
    except URLError as e:
        raise RuntimeError(f"Anthropic URLError: {e}") from e

    content = resp.get("content")
    if not isinstance(content, list) or not content:
        raise RuntimeError(f"Unexpected Anthropic response shape: missing `content`. Keys: {list(resp.keys())}")

    # Anthropic returns content blocks like [{"type":"text","text":"..."}]
    texts: List[str] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str):
            texts.append(block["text"])
    if not texts:
        raise RuntimeError(f"Unexpected Anthropic content blocks: {content!r}")
    return "\n".join(texts).strip()


def _sleep_backoff(attempt: int, base_s: float = 1.0, max_s: float = 20.0) -> None:
    # Exponential backoff with jitter.
    delay = min(max_s, base_s * (2 ** max(0, attempt - 1)))
    delay = delay * (0.7 + 0.6 * random.random())
    time.sleep(delay)


def generate_validated_batch(
    cfg: AnthropicConfig,
    batch_size: int,
    batch_index: int,
    seed: int,
    max_attempts: int,
) -> List[Dict[str, Any]]:
    """
    Generate one batch with retries for:
    - HTTP transient failures
    - JSON parsing errors
    - Validation errors
    """
    last_err: Optional[str] = None
    for attempt in range(1, max_attempts + 1):
        try:
            raw_text = call_anthropic_generate_cases(
                cfg=cfg,
                batch_size=batch_size,
                batch_index=batch_index,
                seed=seed,
                previous_error=last_err,
            )
            parsed = parse_json_strict(raw_text)
            validated = validate_cases(parsed)

            # Enforce uniqueness of patient_id within batch; if duplicates exist, fix deterministically.
            seen: set[str] = set()
            for c in validated:
                pid = c["patient_id"]
                if pid in seen:
                    c["patient_id"] = f"PT-{uuid.uuid4()}"
                seen.add(c["patient_id"])

            if len(validated) != batch_size:
                raise CaseValidationError(f"Expected {batch_size} cases, got {len(validated)}.")
            return validated
        except (json.JSONDecodeError, CaseValidationError, RuntimeError) as e:
            last_err = str(e)
            _eprint(f"[batch {batch_index}] attempt {attempt}/{max_attempts} failed: {last_err}")
            if attempt < max_attempts:
                _sleep_backoff(attempt)
            else:
                raise
    raise RuntimeError("Unreachable: batch generation loop exited unexpectedly.")


def build_config(api_key: str) -> AnthropicConfig:
    # Sensible defaults; override via environment variables.
    model = os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
    max_tokens = int(os.environ.get("ANTHROPIC_MAX_TOKENS", "4096"))
    temperature = float(os.environ.get("ANTHROPIC_TEMPERATURE", "0.2"))
    timeout_s = int(os.environ.get("ANTHROPIC_TIMEOUT_S", "60"))
    return AnthropicConfig(
        api_key=api_key,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        request_timeout_s=timeout_s,
    )


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate synthetic patient cases via Anthropic API.")
    p.add_argument("--count", type=int, default=100, help="Total number of cases to generate.")
    p.add_argument("--batch-size", type=int, default=10, help="Cases per API call.")
    p.add_argument(
        "--out",
        type=str,
        default=os.path.join("data", "patient_cases.json"),
        help="Output JSON file path.",
    )
    p.add_argument(
        "--dotenv",
        type=str,
        default=".env",
        help="Path to .env file containing ANTHROPIC_API_KEY.",
    )
    p.add_argument("--seed", type=int, default=1337, help="Seed for reproducibility of batching/jitter.")
    p.add_argument("--max-attempts", type=int, default=6, help="Max attempts per batch on errors.")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    if args.count <= 0:
        _eprint("`--count` must be > 0.")
        return 2
    if args.batch_size <= 0:
        _eprint("`--batch-size` must be > 0.")
        return 2
    if args.max_attempts <= 0:
        _eprint("`--max-attempts` must be > 0.")
        return 2

    random.seed(args.seed)

    api_key = get_anthropic_api_key(args.dotenv)
    cfg = build_config(api_key)

    total = args.count
    batch_size = args.batch_size
    n_batches = (total + batch_size - 1) // batch_size

    all_cases: List[Dict[str, Any]] = []
    global_seen_ids: set[str] = set()

    for batch_index in range(n_batches):
        remaining = total - len(all_cases)
        this_batch_size = min(batch_size, remaining)
        batch_seed = args.seed + batch_index

        batch_cases = generate_validated_batch(
            cfg=cfg,
            batch_size=this_batch_size,
            batch_index=batch_index,
            seed=batch_seed,
            max_attempts=args.max_attempts,
        )

        # Enforce global uniqueness of patient_id (across batches).
        for c in batch_cases:
            pid = c["patient_id"]
            if pid in global_seen_ids:
                c["patient_id"] = f"PT-{uuid.uuid4()}"
            global_seen_ids.add(c["patient_id"])

        all_cases.extend(batch_cases)
        _eprint(f"Generated {len(all_cases)}/{total} cases...")

    # Final sanity check: validate again to ensure persistence integrity.
    for i, c in enumerate(all_cases):
        try:
            validate_case(c)
        except CaseValidationError as e:
            raise RuntimeError(f"Unexpected invalid case at final validation index {i}: {e}") from e

    ensure_parent_dir(args.out)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(all_cases, f, ensure_ascii=False, indent=2)
        f.write("\n")

    _eprint(f"Wrote {len(all_cases)} cases to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

