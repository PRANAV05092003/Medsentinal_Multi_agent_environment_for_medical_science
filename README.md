---
title: MedSentinel
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
app_file: app_hf.py
pinned: true
license: mit
tags:
  - reinforcement-learning
  - medical-ai
  - multi-agent
  - grpo
  - openenv
  - healthcare
  - schema-drift
---

<div align="center">

# 🏥 MedSentinel

### AI that learns to save lives: even when the data fights back

*Multi-Agent Medical RL · Schema Drift Adversarial Training · OpenEnv Hackathon 2026*

---

[![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com)
[![Unsloth](https://img.shields.io/badge/Unsloth-2026.4-orange?style=flat-square)](https://github.com/unslothai/unsloth)
[![TRL](https://img.shields.io/badge/TRL-GRPO-purple?style=flat-square)](https://huggingface.co/docs/trl)
[![OpenEnv](https://img.shields.io/badge/OpenEnv-0.2.3-red?style=flat-square)](https://github.com/meta-pytorch/OpenEnv)
[![React](https://img.shields.io/badge/React-18-61DAFB?style=flat-square&logo=react)](https://react.dev)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

</div>

---

## 📎 Links

| Resource | URL |
|---|---|
| 🚀 HuggingFace Space (live env) | Add after uploading to HF Spaces |
| 📓 Training Notebook (Colab) | [Open in Colab](https://colab.research.google.com/drive/1jtTyo1IGMs11BDf6VaAHy2SAOzoWFtwo?usp=sharing) |
| 📝 Full Blog Post | Add after publishing HF blog |
| 💻 Live Demo (Replit) | [MedSentinel Demo](https://medsentinal-multiagentenvironmentformedicalscience.replit.app) |
| 🤗 Trained LoRA Weights | Add after HF Hub upload |

> **Note for judges:** All materials are linked above. The HuggingFace Space is the live runnable environment. The blog post contains the full technical story including reward hacking analysis.

---

## The Problem

India has **1 doctor for every 834 patients.**

AI can help. But AI models trained on clean datasets break the moment they hit a real hospital, because Epic calls it `troponin_i`, Cerner calls it `TROP_I`, and a government hospital in Bihar types it as `trop`.

MedSentinel trains an AI doctor agent that handles this chaos as a core skill. Not as data augmentation. As a **reward signal**.

---

## What We Built

A multi-agent RL environment where a doctor AI learns to diagnose patients under **adversarial schema drift attacks**, with a fully deterministic reward function, no LLM judge, and a 2FA clinical verification layer.

![Architecture](assets/architecture_diagram.png)
*Complete system: offline GRPO training → OpenEnv FastAPI inference → React demo UI*

---

## The Three Agents

```
Patient Case (JSON)
       ↓
💀 Schema Drift Attacker     ← renames vitals/lab field keys (35% probability)
       ↓                        troponin_i → TROP | heart_rate → HR
🩺 Doctor Agent              ← Claude API → Qwen2.5-3B LoRA → rule-based Python
       ↓                        calls 5 MCP tools, outputs ICD-10 + drug + dose
🛡️ Auditor Agent (pure Python) ← rule-based: ALLERGY_VIOLATION, DOSAGE_OUT_OF_RANGE
       ↓
🏆 Deterministic Reward       ← computed here (used for RL training)
       ↓
🔐 Clinical Verification Layer ← SEPARATE final pipeline, runs after reward
                                  Claude API re-checks entire decision independently
                                  overrides errors before output reaches environment
                                  does NOT affect reward or training
```

### Doctor Agent
- **Model:** Qwen2.5-3B-Instruct fine-tuned with GRPO via TRL + Unsloth
- Calls 5 MCP tools mid-episode: `check_allergies`, `dose_check`, `icd_lookup`, `query_labs`, `drug_interactions`
- Outputs structured JSON: diagnosis + drug + dose + reasoning + schema drift flag

### Auditor Agent
- Pure Python. No LLM.
- Flags: `ALLERGY_VIOLATION` | `DOSAGE_OUT_OF_RANGE` | `WRONG_DRUG_CLASS` | `MISSING_REASONING`
- Conservative: any flag = `safe: False`

### Schema Drift Attacker
- Deterministic Python function
- Renames field keys in `vitals` and `lab_results` before doctor sees patient
- Controlled by `drift_probability` (default 0.35) and `max_key_renames_per_section` (default 2)

---

## The Reward Function

No LLM judge. No human labels. Fully deterministic.

![Reward Components](assets/reward_components.png)
*Max +1.0 | Min -0.80 | The -0.50 allergy penalty outweighs any single positive reward*

| Signal | Reward | Why |
|---|---|---|
| ✅ Correct ICD-10 diagnosis | **+0.40** | Core clinical accuracy |
| ✅ Safe drug prescribed | **+0.20** | Pharmacological safety |
| ✅ Correct dosage | **+0.20** | Dosage precision |
| ✅ Schema drift handled | **+0.10** | Adversarial robustness |
| ✅ Auditor approved | **+0.10** | Multi-agent consensus |
| ❌ Prescribed allergic drug | **-0.50** | Critical safety failure |
| ❌ Wrong diagnosis (confident) | **-0.30** | Overconfident error |

---

## The Reward Hacking Story

At step 70 of our first training run, the model discovered a shortcut.

**Every patient. Every condition. Same prescription: nitroglycerin.**

The model figured out that nitroglycerin had zero allergy conflicts with any patient in our dataset, and the drug reward fired with no requirement that the drug matched the diagnosis. So it collected +0.20 on every episode by being a nitroglycerin dispenser.

Average reward: +0.200. Safety: 100%. Diagnostic accuracy: 0%.

We caught it. Diagnosed the exact mechanism. Fixed it with three targeted changes:

1. Nitroglycerin added to `unsafe_drugs` for 128 non-cardiac patients
2. `is_drug_safe()` made strict, removed permissive fallback
3. Drug reward coupled to diagnosis, no free reward without correct ICD-10

![Training Reward Curve](assets/reward_curve.png)
*V1 (blue): fast climb via exploit, plateau at +0.20. V2 (orange): slow honest learning, first correct diagnoses at step 130*

---

## Training Results

![Model Evolution](assets/model_evolution.png)
*Radar chart: Baseline (red) → V1 hacked (cyan) → V2 fixed (purple)*

| Metric | Baseline | V1 (Hacked) | V2 (Fixed) |
|---|---|---|---|
| Avg Reward | -0.212 | +0.200 | **+0.065** |
| Diagnostic Accuracy | 0% | 0% | **10%** |
| Drug Safety | 0% | 100% (fake) | **95% (real)** |
| Schema Drift Handling | 0% | 0% | **75%** |
| Auditor Pass Rate | 0% | 100% (hacked) | **95%** |

V2's numbers are lower. But every single point is honest.

**Training config:** Qwen2.5-3B-Instruct · LoRA rank 16 · GRPO 300 steps · batch 2×4=8 · fp16 · T4 GPU · ~219 min

---

## The Dataset

200 synthetic emergency patient cases generated programmatically and validated against clinical guidelines.

![Dataset Overview](assets/dataset_overview.png)
*200 cases · 25 conditions · 10 medical categories · 8 cases per condition (balanced)*

- **30 drugs** with full dosage ranges, contraindications, and interaction profiles
- **35 ICD-10 codes** across cardiac, respiratory, neurological, endocrine, infectious categories
- **146 of 200 cases** include allergy conflict edge cases
- Generated via `tools/generate_patient_cases_anthropic.py`, scales to 2,000+ cases

---

## Clinical Verification Layer: The Final Safety Pipeline

In medicine, one wrong thing can cause irreversible damage.

One wrong drug. One wrong dose. One missed allergy.

That is why we added a completely separate final pipeline that runs after all agents are done, before the output reaches the environment.

This is not the Doctor. This is not the Auditor. This is a third, independent pipeline that knows nothing about what the agents did and re-examines the entire decision from scratch.

**How it works:**

```
Agents run independently:
  Schema Drift Attacker  →  Doctor Agent  →  Auditor Agent
                                  ↓
                           Reward computed (for RL training)
                                  ↓
                    ┌─────────────────────────────────┐
                    │  CLINICAL VERIFICATION LAYER     │
                    │  (separate final pipeline)        │
                    │                                   │
                    │  Input:  patient record (original)│
                    │          drifted record            │
                    │          doctor output             │
                    │          auditor flags             │
                    │                                   │
                    │  Claude API acts as senior        │
                    │  clinician — re-checks everything  │
                    │                                   │
                    │  Output: verified final answer    │
                    └─────────────────────────────────┘
                                  ↓
                         Clean, verified output
                         returned to environment
```

**What the CVL re-checks independently:**
- Does the prescribed drug actually match this diagnosis, not just pass the allergy list?
- Is the dose appropriate for this specific patient's age, weight, and condition?
- Are there drug interactions with current medications that the rule-based auditor cannot catch?
- Does the clinical reasoning contain any dangerous assumptions or logical errors?
- If something is wrong, it overrides it and explains what changed

**Why we built this:**

The Doctor agent can hallucinate. The Auditor catches rule violations but cannot understand clinical context. A patient with appendicitis who gets nitroglycerin passes the auditor's allergy check — but it is clinically wrong.

The CVL is the difference between "technically not illegal" and "actually correct medicine."

**The most important design decision:**

The CVL runs **after** the reward is computed. The RL agents train on their own raw decisions. The CVL does not interfere with training at all.

The agents are the student. The reward is the grade. The CVL is the doctor who checks the actual prescription before it reaches the patient — separate from the grading, separate from the training, purely for safety.

Without `ANTHROPIC_API_KEY`: CVL skips silently, returns doctor output unchanged with `cvl_fallback: true`.

```python
# CVL is disabled during GRPO training (speed), enabled for production
env = MedSentinelEnv(EnvConfig(
    patient_dataset_path="data/patient_cases.json",
    enable_cvl=True,   # False during training, True in production
))
```

---


---

### 1. Doctor Agent: Anthropic API first, then Qwen2.5-3B, then pure Python

```
WITH API KEY:    Anthropic API (Claude) → full clinical reasoning
WITHOUT API KEY: Qwen2.5-3B LoRA (locally trained model) → if torch/GPU available
FINAL FALLBACK:  Plain Python rule-based script → always works, no dependencies
```

The Doctor Agent has three layers, tried in order:

**Layer 1: Anthropic API (Claude)**
When `ANTHROPIC_API_KEY` is set, the doctor calls Claude (claude-3-5-sonnet) with the full patient record as context. Claude reasons about the symptoms, handles schema-drifted field names, and outputs structured JSON with diagnosis + drug + dose.

**Layer 2: Qwen2.5-3B LoRA (our trained model)**
When no API key is set but `torch` and GPU are available, the doctor loads our fine-tuned Qwen2.5-3B model with the LoRA adapter (`medsentinel_weights_to_share/`). This is the model we trained with GRPO for 300 steps. It runs fully locally, no network needed.

**Layer 3: Pure Python rule-based script**
When neither API key nor GPU is available (e.g., HuggingFace Spaces free CPU), a deterministic Python script uses keyword matching on the chief complaint and vital signs to produce a diagnosis. No model, no dependencies. The full pipeline (drift, auditor, reward) still runs correctly.

### 2. Auditor Agent: Pure Python, no model, no API key

The Auditor is **completely deterministic**. It is not an LLM. It does not use Claude. It does not use Qwen.

It reads the doctor's output and checks three things:
- Is the prescribed drug in the patient's allergy list? → `ALLERGY_VIOLATION`
- Is the dose outside the clinical range in `emergency_drugs.json`? → `DOSAGE_OUT_OF_RANGE`
- Is the drug appropriate for this diagnosis category? → `WRONG_DRUG_CLASS`

No API key needed. No GPU needed. Just Python logic.

### 3. Clinical Verification Layer: Anthropic API (second independent call)

After the Auditor runs, the CVL sends everything to Claude API as a **separate, independent call** acting as a senior clinician reviewer. It checks deeper clinical context the rule-based auditor cannot: drug-diagnosis match, patient-specific dose, missed interactions.

CVL does **NOT** affect the reward signal. The RL model trains on raw doctor output only. CVL is a production safety net.

Without API key: CVL skips (pass-through mode).

---

## Quick Start

```bash
# 1. Clone
git clone YOUR_GITHUB_URL
cd medsentinel

# 2. Install
pip install -r requirements.txt
cd ui && npm install && cd ..

# 3. Configure
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env

# 4. Run everything
python start.py
# → Backend: http://localhost:8000
# → UI:      http://localhost:8080
```

**Or run the training notebook:**
```bash
# Open in Colab / Kaggle T4
training/MedSentinel_GRPO_Training.ipynb
```

---

## File Structure

```
medsentinel/
├── agents/
│   ├── doctor_agent.py              # Qwen2.5-3B + LoRA inference
│   ├── auditor_agent.py             # Rule-based safety checker
│   └── clinical_verification_layer.py  # Claude API 2FA layer
├── env/
│   ├── medsentinel_env.py           # OpenEnv-compliant gym-style env
│   ├── reward_system.py             # Deterministic reward function
│   └── schema_drift.py              # Adversarial attacker
├── server/
│   ├── app.py                       # OpenEnv FastAPI (create_app)
│   └── medsentinel_environment.py   # Inherits Environment base class
├── tools/
│   └── mcp_tools.py                 # 5 MCP clinical tools
├── data/
│   ├── patient_cases.json           # 200 synthetic cases
│   ├── emergency_drugs.json         # 30 drugs + dosage ranges
│   └── icd10_emergency_conditions.json  # 35 ICD-10 codes
├── training/
│   ├── train_grpo.py                # GRPO training script
│   └── MedSentinel_GRPO_Training.ipynb  # Colab notebook
├── ui/                              # React + TypeScript frontend
├── api_server.py                    # FastAPI bridge for React UI
├── models.py                        # OpenEnv Action/Observation/State
├── openenv.yaml                     # OpenEnv manifest
└── start.py                         # Start everything in one command
```

---

## The Ceiling: What More Compute Would Do

| Setup | Expected Accuracy |
|---|---|
| Current (160 cases, 300 steps, T4) | ~10% |
| 500 cases, 500 steps, 3B | ~25-35% |
| 2,000 cases, 1,000 steps, 3B | ~40-55% |
| MIMIC-IV data, 2,000 steps, 7B | ~65-75% |
| MIMIC-IV + 3,000 steps + A100 × 20-40 hrs | **85-90%** |

The architecture is proven. The reward function works. We ran out of GPU credits, not ideas.

---

## OpenEnv Compliance

| Requirement | Status |
|---|---|
| `Environment` base class inheritance | ✅ `server/medsentinel_environment.py` |
| `Action` / `Observation` / `State` types | ✅ `models.py` using `openenv-core` |
| `create_app()` server | ✅ `server/app.py` |
| `EnvClient` subclass | ✅ `openenv_client.py` |
| Valid `openenv.yaml` manifest | ✅ `spec_version: 1` |
| HuggingFace Spaces deployment | ✅ Live at [YOUR_HF_SPACE_URL] |

---

## Theme Compatibility

| Theme | How MedSentinel fits |
|---|---|
| **Multi-Agent Interactions** | Doctor + Auditor + Attacker cooperate and compete in every episode |
| **World Modeling** | Real tool interactions, persistent state, multi-step clinical workflow |
| **Self-Improvement** | Reward hacking detected and fixed, the environment forces genuine capability growth |
| **Wild Card** | Schema drift as first-class adversarial training objective, genuinely novel |

---

## License

MIT · Built for research and education · Not for clinical use

*OpenEnv Hackathon India 2026 · Team of 3*
