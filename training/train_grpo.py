#!/usr/bin/env python3
"""
MedSentinel GRPO Training Script
=================================

Trains a local LLM (Qwen2.5-3B-Instruct) using Group Relative Policy Optimization
(GRPO) via TRL + Unsloth on the MedSentinel RL environment.

Designed to run on:
- Kaggle (free T4 GPU, 16GB VRAM)
- Google Colab Pro (T4/A100)
- Any GPU with 16GB+ VRAM

Training flow:
1. Load base Qwen2.5-3B-Instruct with Unsloth 4-bit quantization
2. Format patient cases as GRPO prompt/completion pairs
3. Define reward_fn() wrapping compute_reward()
4. Run GRPOTrainer for N steps
5. Save checkpoints + reward curve plot
6. Run before/after eval comparison
7. Push to HuggingFace Hub (optional)

Usage:
  python training/train_grpo.py
  python training/train_grpo.py --steps 500 --push-to-hub

Kaggle usage:
  # Add to Kaggle notebook cell:
  !pip install unsloth trl transformers datasets -q
  !python training/train_grpo.py --steps 200 --save-path /kaggle/working/medsentinel_model
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure repo root on path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from env.schema_drift import apply_schema_drift


# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

DEFAULT_BASE_MODEL = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"
DEFAULT_DATASET_PATH = os.path.join(REPO_ROOT, "data", "patient_cases.json")
DEFAULT_SAVE_PATH = os.path.join(REPO_ROOT, "checkpoints", "medsentinel_grpo")
DEFAULT_STEPS = 500
DEFAULT_BATCH_SIZE = 4
DEFAULT_LR = 5e-6
DEFAULT_MAX_SEQ_LEN = 1024
DEFAULT_NUM_GENERATIONS = 4  # GRPO group size


# ─────────────────────────────────────────────────────────────
# Prompt formatting
# ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are MedSentinel's Doctor agent in a simulated emergency department.
Your task: analyze the patient record and provide a diagnosis and treatment.

OUTPUT FORMAT — Return ONLY valid JSON with exactly these keys:
{
  "reasoning": "<clinical reasoning>",
  "diagnosis_icd10": "<ICD-10 code>",
  "diagnosis_name": "<condition name>",
  "prescribed_drug": "<drug name>",
  "dosage_mg": <number or null>,
  "confidence": <0.0 to 1.0>,
  "schema_drift_handled": <true or false>
}

RULES:
- reasoning must be at least 50 words
- diagnosis_icd10 must be a valid ICD-10 code
- prescribed_drug must not match any known_allergies
- dosage_mg must be within clinical range or null
- schema_drift_handled = true if you detected renamed keys in vitals/labs
- Return ONLY the JSON object. No markdown, no prose, no code fences."""


def format_patient_prompt(patient: Dict[str, Any]) -> str:
    """Format a patient case as a model prompt."""
    patient_clean = {k: v for k, v in patient.items()
                     if k not in ("ground_truth_diagnosis", "safe_drugs", "unsafe_drugs")}
    patient_json = json.dumps(patient_clean, indent=2)
    return f"Patient record:\n{patient_json}\n\nProvide your diagnosis and treatment in the required JSON format."


def format_for_grpo(
    patient: Dict[str, Any],
    apply_drift: bool = False,
    drift_seed: int = 42,
) -> Dict[str, str]:
    """Format a single patient case as a GRPO training example."""
    if apply_drift:
        drifted_patient, drift_occurred, _ = apply_schema_drift(
            patient,
            seed=drift_seed,
            drift_probability=1.0,
            max_key_renames_per_section=2,
        )
    else:
        drifted_patient = patient
        drift_occurred = False

    return {
        "prompt": format_patient_prompt(drifted_patient),
        "patient_json": json.dumps(patient),
        "drift_occurred": str(drift_occurred),
    }


# ─────────────────────────────────────────────────────────────
# Reward function (wraps compute_reward for GRPO)
# ─────────────────────────────────────────────────────────────

def _parse_model_output(text: str) -> Dict[str, Any]:
    """Extract JSON from model output. Returns empty dict on failure."""
    import re
    text = text.strip()
    # Try direct parse
    try:
        return json.loads(text)
    except Exception:
        pass
    # Find JSON block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass
    return {}


def build_reward_fn(patients_by_id: Dict[str, Dict[str, Any]]):
    """
    Returns a GRPO-compatible reward function.

    GRPO reward_fn signature: fn(prompts, completions, **kwargs) -> list[float]
    - prompts: list of input strings
    - completions: list of model output strings
    - kwargs: may include 'patient_json' passed from dataset

    Returns list of floats (one per completion).
    """
    from env.reward_system import compute_reward
    from agents.auditor_agent import audit_doctor_output

    def reward_fn(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
        rewards = []
        patient_jsons = kwargs.get("patient_json", [None] * len(completions))
        drift_flags = kwargs.get("drift_occurred", ["False"] * len(completions))

        for completion, patient_json_str, drift_str in zip(completions, patient_jsons, drift_flags):
            try:
                patient = json.loads(patient_json_str) if patient_json_str else {}
                doctor_output = _parse_model_output(completion)
                drift_flag = str(drift_str).lower() == "true"

                if not doctor_output:
                    rewards.append(-0.5)
                    continue

                auditor = audit_doctor_output(doctor_output, patient)
                auditor_flags = {
                    "is_correct": bool(auditor.get("safe", False)),
                    "flags": auditor.get("flags", []),
                }

                reward, _ = compute_reward(
                    doctor_output,
                    patient,
                    auditor_flags=auditor_flags,
                    drift_flag=drift_flag,
                )
                rewards.append(float(reward))
            except Exception as e:
                rewards.append(-0.3)

        return rewards

    return reward_fn


# ─────────────────────────────────────────────────────────────
# Dataset loader
# ─────────────────────────────────────────────────────────────

def load_grpo_dataset(dataset_path: str, test_fraction: float = 0.2, seed: int = 42):
    """
    Load patient cases and split into train/test HuggingFace Datasets.
    """
    try:
        from datasets import Dataset
    except ImportError:
        raise ImportError("Install `datasets`: pip install datasets")

    with open(dataset_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    import random
    rng = random.Random(seed)
    rng.shuffle(cases)
    split = max(1, int(len(cases) * (1 - test_fraction)))
    train_cases = cases[:split]
    test_cases = cases[split:]

    import random as _random
    rng2 = _random.Random(seed + 1)

    train_data = []
    for i, c in enumerate(train_cases):
        apply_drift = rng2.random() < 0.5
        train_data.append(
            format_for_grpo(
                c,
                apply_drift=apply_drift,
                drift_seed=seed + i,
            )
        )

    test_data = [format_for_grpo(c) for c in test_cases]

    print("[train] First 5 train drift flags:", [ex.get("drift_occurred", "False") for ex in train_data[:5]])

    train_ds = Dataset.from_list(train_data)
    test_ds = Dataset.from_list(test_data)
    return train_ds, test_ds, cases


# ─────────────────────────────────────────────────────────────
# Model loader (Unsloth)
# ─────────────────────────────────────────────────────────────

def load_model_unsloth(
    model_name: str = DEFAULT_BASE_MODEL,
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
):
    """
    Load model + tokenizer with Unsloth 4-bit quantization.
    Falls back to plain HuggingFace transformers if Unsloth not available.
    """
    try:
        from unsloth import FastLanguageModel
        print(f"[train] Loading {model_name} via Unsloth (4-bit)...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_len,
            dtype=None,
            load_in_4bit=True,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                             "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )
        print("[train] Unsloth model loaded with LoRA adapters.")
        return model, tokenizer, "unsloth"
    except ImportError:
        print("[train] Unsloth not found. Falling back to HuggingFace transformers...")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        print("[train] Transformers model loaded.")
        return model, tokenizer, "transformers"


# ─────────────────────────────────────────────────────────────
# Baseline evaluation (before training)
# ─────────────────────────────────────────────────────────────

def run_baseline_eval(
    model,
    tokenizer,
    test_cases: List[Dict[str, Any]],
    n_cases: int = 20,
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Quick baseline eval: run model on n_cases, compute reward stats.
    Returns {'avg_reward': float, 'accuracy': float, 'safety': float}
    """
    import torch
    from env.reward_system import compute_reward, is_diagnosis_correct
    from agents.auditor_agent import audit_doctor_output

    model.eval()
    rewards, correct, safe = [], 0, 0
    eval_cases = test_cases[:n_cases]

    for patient in eval_cases:
        prompt = format_patient_prompt(patient)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        try:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.2,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
            completion = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            doctor_output = _parse_model_output(completion)

            if doctor_output:
                auditor = audit_doctor_output(doctor_output, patient)
                auditor_flags = {"is_correct": auditor.get("safe", False), "flags": auditor.get("flags", [])}
                reward, _ = compute_reward(doctor_output, patient, auditor_flags=auditor_flags)
                rewards.append(reward)

                predicted = doctor_output.get("diagnosis_icd10", "") or doctor_output.get("diagnosis_name", "")
                if is_diagnosis_correct(str(predicted), str(patient.get("ground_truth_diagnosis", ""))):
                    correct += 1
                if auditor.get("safe", False):
                    safe += 1
            else:
                rewards.append(-0.5)
        except Exception as e:
            rewards.append(-0.3)

    n = max(1, len(rewards))
    return {
        "avg_reward": sum(rewards) / n,
        "accuracy_pct": 100.0 * correct / len(eval_cases),
        "safety_pct": 100.0 * safe / len(eval_cases),
        "n_evaluated": len(eval_cases),
    }


# ─────────────────────────────────────────────────────────────
# Save reward curve plot
# ─────────────────────────────────────────────────────────────

def save_reward_curve(reward_log: List[float], save_path: str) -> str:
    """Save a matplotlib reward curve PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        steps = list(range(1, len(reward_log) + 1))
        # Smoothed (moving average, window 10)
        window = min(10, len(reward_log))
        smoothed = np.convolve(reward_log, np.ones(window) / window, mode="valid")

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(steps, reward_log, alpha=0.3, color="steelblue", label="Raw reward")
        ax.plot(steps[window - 1:], smoothed, color="steelblue", linewidth=2, label=f"Smoothed (w={window})")
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Average Reward")
        ax.set_title("MedSentinel GRPO Training — Reward Curve")
        ax.legend()
        ax.grid(alpha=0.3)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[train] Reward curve saved to {save_path}")
        return save_path
    except ImportError:
        print("[train] matplotlib not available, skipping reward curve plot.")
        return ""


# ─────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────

def train(
    *,
    base_model: str = DEFAULT_BASE_MODEL,
    dataset_path: str = DEFAULT_DATASET_PATH,
    save_path: str = DEFAULT_SAVE_PATH,
    steps: int = DEFAULT_STEPS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
    num_generations: int = DEFAULT_NUM_GENERATIONS,
    push_to_hub: bool = False,
    hub_model_id: Optional[str] = None,
    seed: int = 42,
):
    print("\n" + "="*60)
    print("  MedSentinel GRPO Training")
    print("="*60)
    print(f"  Base model  : {base_model}")
    print(f"  Dataset     : {dataset_path}")
    print(f"  Steps       : {steps}")
    print(f"  Batch size  : {batch_size}")
    print(f"  LR          : {lr}")
    print(f"  Save path   : {save_path}")
    print("="*60 + "\n")

    try:
        from trl import GRPOConfig, GRPOTrainer
    except ImportError:
        raise ImportError("Install TRL: pip install trl>=0.8.0")

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] Device: {device}")
    if device == "cpu":
        print("[train] WARNING: Training on CPU will be very slow. Use GPU for real training.")

    # 1. Load dataset
    print("[train] Loading dataset...")
    train_ds, test_ds, all_cases = load_grpo_dataset(dataset_path, seed=seed)
    test_cases = [json.loads(r["patient_json"]) for r in test_ds]
    print(f"[train] Train: {len(train_ds)} cases | Test: {len(test_ds)} cases")

    # 2. Load model
    model, tokenizer, backend = load_model_unsloth(base_model, max_seq_len)

    # 3. Baseline eval
    print("\n[train] Running BASELINE evaluation (before training)...")
    try:
        baseline = run_baseline_eval(model, tokenizer, test_cases, n_cases=min(20, len(test_cases)), device=device)
        print(f"[train] BASELINE — Reward: {baseline['avg_reward']:.3f} | "
              f"Accuracy: {baseline['accuracy_pct']:.1f}% | Safety: {baseline['safety_pct']:.1f}%")
    except Exception as e:
        print(f"[train] Baseline eval failed (non-fatal): {e}")
        baseline = {"avg_reward": 0.0, "accuracy_pct": 0.0, "safety_pct": 0.0}

    # 4. Build reward function
    patients_by_id = {c["patient_id"]: c for c in all_cases}
    reward_fn = build_reward_fn(patients_by_id)

    # 5. Configure GRPO
    grpo_config = GRPOConfig(
        output_dir=save_path,
        num_train_epochs=1,
        max_steps=steps,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=max(1, 8 // batch_size),
        learning_rate=lr,
        num_generations=num_generations,
        max_prompt_length=max_seq_len // 2,
        max_completion_length=512,
        temperature=0.9,
        logging_steps=10,
        save_steps=100,
        seed=seed,
        report_to="none",
        remove_unused_columns=False,
    )

    # 6. Format dataset with system prompt applied
    def format_with_system(example):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["prompt"]},
        ]
        return {
            "prompt": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
            "patient_json": example["patient_json"],
            "drift_occurred": example.get("drift_occurred", "False"),
        }

    print("[train] Formatting dataset with chat template...")
    train_ds_formatted = train_ds.map(format_with_system)

    # 7. Train
    print(f"\n[train] Starting GRPO training for {steps} steps...")
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward_fn],
        args=grpo_config,
        train_dataset=train_ds_formatted,
        processing_class=tokenizer,
    )

    reward_log: List[float] = []

    # Hook to capture reward log
    original_log = trainer.log
    def patched_log(logs: Dict[str, Any], *args, **kwargs):
        if "rewards/mean" in logs:
            reward_log.append(logs["rewards/mean"])
        elif "reward" in logs:
            reward_log.append(logs["reward"])
        original_log(logs, *args, **kwargs)
    trainer.log = patched_log

    start_time = time.time()
    train_result = trainer.train()
    elapsed = time.time() - start_time
    print(f"\n[train] Training complete in {elapsed/60:.1f} minutes.")

    # 8. Save model
    os.makedirs(save_path, exist_ok=True)
    print(f"[train] Saving model to {save_path}...")
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)

    # 9. Save reward curve
    curve_path = os.path.join(REPO_ROOT, "dashboard", "static", "reward_curve.png")
    if reward_log:
        save_reward_curve(reward_log, curve_path)
    else:
        print("[train] No reward log captured (check trainer logging config).")

    # 10. Post-training eval
    print("\n[train] Running POST-TRAINING evaluation...")
    try:
        if backend == "unsloth":
            from unsloth import FastLanguageModel
            FastLanguageModel.for_inference(model)
        post = run_baseline_eval(model, tokenizer, test_cases, n_cases=min(20, len(test_cases)), device=device)
        print(f"[train] POST-TRAIN — Reward: {post['avg_reward']:.3f} | "
              f"Accuracy: {post['accuracy_pct']:.1f}% | Safety: {post['safety_pct']:.1f}%")
        improvement = post['avg_reward'] - baseline['avg_reward']
        print(f"\n[train] IMPROVEMENT: {improvement:+.3f} reward | "
              f"{post['accuracy_pct'] - baseline['accuracy_pct']:+.1f}% accuracy")
    except Exception as e:
        print(f"[train] Post-training eval failed: {e}")
        post = baseline

    # 11. Save comparison summary
    summary = {
        "baseline": baseline,
        "post_training": post,
        "training_steps": steps,
        "base_model": base_model,
        "reward_log": reward_log,
        "elapsed_minutes": elapsed / 60,
    }
    summary_path = os.path.join(save_path, "training_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[train] Summary saved to {summary_path}")

    # 12. Push to HuggingFace Hub (optional)
    if push_to_hub and hub_model_id:
        print(f"\n[train] Pushing to HuggingFace Hub: {hub_model_id}")
        try:
            model.push_to_hub(hub_model_id)
            tokenizer.push_to_hub(hub_model_id)
            print(f"[train] Model pushed to https://huggingface.co/{hub_model_id}")
        except Exception as e:
            print(f"[train] Hub push failed: {e}")

    print("\n" + "="*60)
    print("  TRAINING COMPLETE")
    print(f"  Baseline accuracy  : {baseline['accuracy_pct']:.1f}%")
    print(f"  Post-train accuracy: {post['accuracy_pct']:.1f}%")
    print(f"  Model saved to     : {save_path}")
    print("="*60)

    return summary


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Train MedSentinel with GRPO")
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL, help="HuggingFace model ID or local path")
    p.add_argument("--dataset", default=DEFAULT_DATASET_PATH, help="Path to patient_cases.json")
    p.add_argument("--save-path", default=DEFAULT_SAVE_PATH, help="Where to save the trained model")
    p.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="Training steps")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size per device")
    p.add_argument("--lr", type=float, default=DEFAULT_LR, help="Learning rate")
    p.add_argument("--num-generations", type=int, default=DEFAULT_NUM_GENERATIONS, help="GRPO group size")
    p.add_argument("--max-seq-len", type=int, default=DEFAULT_MAX_SEQ_LEN, help="Max sequence length")
    p.add_argument("--push-to-hub", action="store_true", help="Push trained model to HuggingFace Hub")
    p.add_argument("--hub-model-id", default=None, help="HuggingFace Hub model ID (e.g. username/medsentinel-7b)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(
        base_model=args.base_model,
        dataset_path=args.dataset,
        save_path=args.save_path,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        num_generations=args.num_generations,
        max_seq_len=args.max_seq_len,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        seed=args.seed,
    )
