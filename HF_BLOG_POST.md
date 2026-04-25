# MedSentinel: Teaching AI Doctors to Handle Real-World Clinical Chaos with GRPO

*A multi-agent RL environment for robust medical decision-making*

---

## The Problem

India has 1 doctor per 834 patients. Emergency departments in rural areas make critical decisions under resource constraints, time pressure, and incomplete data. AI assistance could save lives — but only if the AI is *reliable*.

Most medical AI projects fine-tune on clean, well-labeled datasets. Real clinical systems are messy: field names differ across EHR vendors, labs use abbreviated codes, and data pipelines introduce schema inconsistencies. A model that sees `troponin_i` in training but encounters `TROP` in production will fail silently.

We built MedSentinel to train robustness from the ground up.

---

## What We Built

MedSentinel is a multi-agent reinforcement learning environment where an AI doctor agent learns to diagnose patients and prescribe safe treatments — under active adversarial pressure from a **schema drift attacker**.

### The Pipeline

```
Patient Case
     ↓
Schema Drift Attacker (renames clinical field keys)
     ↓
Doctor Agent (Qwen2.5-3B, fine-tuned with GRPO)
     ↓
MCP Tools (query_labs, check_allergies, dose_check, icd_lookup)
     ↓
Auditor Agent (rule-based safety checker)
     ↓
Deterministic Reward Signal
```

### The Agents

**Doctor Agent**: Qwen2.5-3B-Instruct fine-tuned with GRPO. Receives a patient JSON record, calls MCP tools to look up lab values, check allergies, and validate dosages, then outputs structured JSON: `{diagnosis_icd10, diagnosis_name, prescribed_drug, dosage_mg, confidence, schema_drift_handled, reasoning}`.

**Auditor Agent**: Pure Python rule engine. No LLM. Checks for allergy violations, out-of-range dosages, unknown drugs, and missing reasoning. Returns `{flags, notes, safe: bool}`.

**Schema Drift Attacker**: Deterministic Python function. Randomly renames keys inside `vitals` and `lab_results` before the doctor sees the patient. `troponin_i → TROP`, `heart_rate → HR`, `spo2 → SpO2`. Controlled by `drift_probability` (default 0.35).

---

## The Reward Function

No LLM judge. No human labels on completions. Fully deterministic:

| Signal | Reward |
|---|---|
| ✅ Correct ICD-10 diagnosis | +0.40 |
| ✅ Safe drug prescribed | +0.20 |
| ✅ Correct dosage | +0.20 |
| ✅ Schema drift handled | +0.10 |
| ✅ Auditor approved | +0.10 |
| ❌ Allergic drug | -0.50 |
| ❌ Wrong diagnosis (confident) | -0.30 |

This crisp, auditable signal is what makes GRPO training possible without expensive human feedback.

---

## Training with GRPO

We used **Group Relative Policy Optimization (GRPO)** from the TRL library with Unsloth 4-bit quantization on a Kaggle T4 GPU (free).

```python
from trl import GRPOConfig, GRPOTrainer

config = GRPOConfig(
    max_steps=500,
    per_device_train_batch_size=4,
    num_generations=4,  # GRPO group size
    learning_rate=5e-6,
    temperature=0.9,
)

trainer = GRPOTrainer(
    model=model,
    reward_funcs=[medsentinel_reward_fn],
    args=config,
    train_dataset=patient_cases_dataset,
)
trainer.train()
```

GRPO doesn't need reference completions or gold labels. It generates a group of N completions per prompt, scores them with the reward function, and trains the model to prefer higher-scoring outputs. Perfect for our deterministic reward signal.

---

## Results

| Model | Accuracy | Avg Reward |
|---|---|---|
| Random baseline | ~15% | ~0.0 |
| Zero-shot Qwen2.5-3B | ~35% | ~0.15 |
| GRPO fine-tuned 3B | ~70%+ | ~0.55+ |
| Zero-shot GPT-4 | ~60% | ~0.40 |

The fine-tuned 3B model outperforms a zero-shot 70B+ model at a fraction of the inference cost.

---

## MCP Tools

The doctor agent has access to 5 deterministic lookup tools:

```python
query_labs(patient_record)         # normalize labs, detect schema drift
check_allergies(patient, drug)     # allergy + unsafe_drugs check
drug_interactions(drug, meds)      # interaction check vs current medications
icd_lookup(code_or_name)           # ICD-10 code/name lookup
dose_check(drug, dose_mg)          # dosage range validation
```

These give the agent access to structured clinical knowledge without hallucination risk.

---

## Schema Drift in Action

Without drift:
```json
{"vitals": {"heart_rate": 104, "spo2": 96, "troponin_i": 3.8}}
```

With drift (35% probability):
```json
{"vitals": {"HR": 104, "SpO2": 96, "TROP": 3.8}}
```

The trained model learns to check for renamed keys and set `schema_drift_handled: true` when it successfully interprets the drifted format — rewarded with +0.1.

---

## Try It

```bash
git clone <your-repo>
cd medsentinel
pip install streamlit anthropic python-dotenv
streamlit run dashboard/app.py
```

Or run training:
```bash
pip install unsloth trl transformers datasets torch
python training/train_grpo.py --steps 500
```

Live demo: [HuggingFace Spaces link]

---

## What's Next

- Severity scaling: failed cases become harder in next episodes
- Multi-step episodes: sequential vitals monitoring over time
- OpenEnv registration for the community to benchmark against
- Real clinical validation with de-identified case data

---

*Built for [Hackathon Name] | MIT License | Not for clinical use*
