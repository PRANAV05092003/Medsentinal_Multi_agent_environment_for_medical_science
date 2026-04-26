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

*Multi-Agent Medical RL · Schema Drift Adversarial Training · OpenEnv Hackathon India 2026*
<p align="center">
  <img src="assets/assets/WhatsApp Image 2026-04-26 at 9.11.00 AM.jpeg" width="100%" alt="med" />
</p>
---

[![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com)
[![Unsloth](https://img.shields.io/badge/Unsloth-GRPO-orange?style=flat-square)](https://github.com/unslothai/unsloth)
[![TRL](https://img.shields.io/badge/TRL-GRPO-purple?style=flat-square)](https://huggingface.co/docs/trl)
[![OpenEnv](https://img.shields.io/badge/OpenEnv-0.2.3-red?style=flat-square)](https://github.com/meta-pytorch/OpenEnv)
[![React](https://img.shields.io/badge/React-18-61DAFB?style=flat-square&logo=react)](https://react.dev)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

</div>

---

## Links

| Resource | URL |
|---|---|
| 🚀 HuggingFace Space (HF Code) | [Open in HF](https://huggingface.co/spaces/PRANAV05092003/Medsentinal/tree/main) |
| 🚀 HuggingFace Space (live env) | [Live Demo](https://huggingface.co/spaces/PRANAV05092003/Medsentinal) |
| 📓 Training Notebook (Colab) | [Open in Colab](https://colab.research.google.com/drive/1jtTyo1IGMs11BDf6VaAHy2SAOzoWFtwo?usp=sharing) |
| 📝 Full Blog Post | [Blog.md](https://huggingface.co/spaces/PRANAV05092003/Medsentinal/blob/main/Blog.md) |
| 💻 Live Demo (Replit) | [MedSentinel Demo](https://medsentinal-multiagentenvironmentformedicalscience.replit.app) |
| 🤗 Trained LoRA Weights |[ Weights ](https://drive.google.com/drive/folders/18f1pBvxgHi5ZnuKYgOvme4WyzYWIlBHx?usp=sharing)|

> **Note for judges:** The HuggingFace Space is the live runnable environment. `Blog.md` contains the full technical story including the reward hacking discovery, fix, and results.

---

## The Problem

India has **1 doctor for every 834 patients.**

AI can help. But AI models trained on clean datasets break the moment they hit a real hospital, because Epic calls it `troponin_i`, Cerner calls it `TROP_I`, and a government hospital in Bihar types it as `trop`.

MedSentinel trains an AI doctor agent that handles this chaos as a core skill. Not as data augmentation. As a **reward signal**.

---

## What We Built

A multi-agent RL environment where a doctor AI learns to diagnose patients under **adversarial schema drift attacks**, with a fully deterministic reward function, no LLM judge, and a separate Clinical Verification Layer as a final safety pipeline.

---

## The Three Agents

```
Patient Case (JSON)
       |
💀 Schema Drift Attacker     <- pure Python, renames vitals/lab keys (35% probability)
       |                        troponin_i -> TROP | heart_rate -> HR | spo2 -> SpO2
       |
🩺 Doctor Agent              <- Layer 1: Claude API (primary, when ANTHROPIC_API_KEY set)
       |                        Layer 2: Qwen2.5-3B LoRA (when GPU available, no key)
       |                        Layer 3: rule-based Python (always works, no deps)
       |                        calls 5 MCP tools, outputs ICD-10 + drug + dose
       |
🛡️ Auditor Agent             <- pure Python only, no LLM, no model, no API
       |                        ALLERGY_VIOLATION | DOSAGE_OUT_OF_RANGE | WRONG_DRUG_CLASS
       |
🏆 Deterministic Reward      <- computed here, used for RL training
       |
🔐 CVL (Claude API)          <- SEPARATE final pipeline, runs after reward is computed
                                 re-checks everything independently
                                 does NOT affect reward or training


```
## Agent Details

### Doctor Agent

The Doctor has three layers tried in order:

**Layer 1: Anthropic API (Claude)**: when `ANTHROPIC_API_KEY` is set, calls `claude-3-5-sonnet-20241022` with the full patient record. Handles schema-drifted field names, outputs strict JSON. This is what runs on HuggingFace Spaces.

**Layer 2: Qwen2.5-3B LoRA (our trained model)**: when no API key but torch and GPU are available, loads `unsloth/qwen2.5-3b-instruct-bnb-4bit` with our LoRA adapter from `medsentinel_weights_to_share/`. This is the model trained with GRPO for 300 steps on Google Colab T4. Runs fully locally.

**Layer 3: Pure Python rule-based script**: when neither API key nor GPU is available, a deterministic keyword-matching and scoring script diagnoses using vitals and labs. No model, no dependencies. Full pipeline still runs.

Output contract (all three layers return the same shape):
```json
{
  "diagnosis_icd10": "I21.9",
  "diagnosis_name": "STEMI",
  "prescribed_drug": "nitroglycerin",
  "dosage_mg": 0.4,
  "confidence": 0.85,
  "schema_drift_handled": true,
  "reasoning": "Elevated troponin 3.8 ng/mL + crushing chest pain..."
}
```

### Auditor Agent

**Pure Python. No LLM. No model. No API key.**

Reads the doctor output and checks three things against `data/emergency_drugs.json`:
- Is the prescribed drug in the patient's allergy list? `ALLERGY_VIOLATION`
- Is the dose outside clinical min/max range? `DOSAGE_OUT_OF_RANGE`
- Is the drug appropriate for this diagnosis category? `WRONG_DRUG_CLASS`

Conservative: any flag = `safe: False`. Fully deterministic, same input always gives same output.

### Schema Drift Attacker

**Pure Python. No model. No API.**

Renames field keys in `vitals` and `lab_results` before the doctor sees the patient. Controlled by `drift_probability` (default 0.35) and `max_key_renames_per_section` (default 2).

### MCP Tools (5 tools)

```python
query_labs(patient_record)          # normalize labs, detect schema drift aliases
check_allergies(patient, drug)      # allergy + unsafe_drugs check
drug_interactions(drug, meds)       # interaction check vs current medications
icd_lookup(code_or_name)            # ICD-10 code/name lookup
dose_check(drug, dose_mg)           # dosage range validation
```

---

## The Reward Function

No LLM judge. No human labels. Fully deterministic. Zero prompt injection risk.

<p align="center">
  <img src="assets/assets/medsentinel_reward_components.png" width="100%" alt="Reward component weights and penalty structure" />
</p>
<p align="center"><i>Max +1.0 | Min -0.80 | The -0.50 allergy penalty outweighs any single positive reward</i></p>

| Signal | Reward | Why |
|---|---|---|
| Correct ICD-10 diagnosis | **+0.40** | Core clinical accuracy |
| Safe drug prescribed | **+0.20** | Pharmacological safety |
| Correct dosage | **+0.20** | Dosage precision |
| Schema drift handled | **+0.10** | Adversarial robustness |
| Auditor approved | **+0.10** | Multi-agent consensus |
| Prescribed allergic drug | **-0.50** | Critical safety failure |
| Wrong diagnosis (confident) | **-0.30** | Overconfident error |


---

## The Reward Hacking Story

At step 70 of our first training run the model discovered a shortcut.

**Every patient. Every condition. Same prescription: nitroglycerin.**

Average reward: +0.200. Safety: 100%. Diagnostic accuracy: 0%.

The model figured out nitroglycerin had zero allergy conflicts with any patient in the dataset. The drug reward fired with no requirement the drug matched the diagnosis. Guaranteed +0.20 every episode by being a nitroglycerin dispenser.

We caught it. Fixed it with three targeted changes verified in `training_summary.json`:

1. Nitroglycerin added to `unsafe_drugs` for 128 non-cardiac patients
2. `is_drug_safe()` made strict, removed permissive fallback
3. Drug reward coupled to diagnosis, fires only when diagnosis is also correct

<p align="center">
  <img src="assets/assets/medsentinel_reward_curve.png" width="100%" alt="GRPO training reward curve V1 vs V2" />
</p>
<p align="center"><i>V1 (blue): fast climb via exploit, plateau at +0.20. V2 (orange): slow honest learning, first correct diagnoses at step 130</i></p>

---

## Training Results

<p align="center">
  <img src="assets/assets/1777144749858.png" width="100%" alt="Model evolution dashboard radar chart" />
</p>
<p align="center"><i>Radar chart: Baseline (red) → V1 hacked (cyan) → V2 fixed (purple)</i></p>

| Metric | Baseline | V1 Hacked | V2 Fixed |
|---|---|---|---|
| Avg Reward | -0.212 | +0.200 | **+0.065** |
| Diagnostic Accuracy | 0% | 0% | **10%** |
| Drug Safety | 0% | 100% (fake) | **95% (real)** |
| Schema Drift Handling | 0% | 0% | **75%** |
| Auditor Pass Rate | 0% | 100% (hacked) | **95%** |

V2 numbers are lower. Every single point is honest.

**Training config:**
- Base model: `unsloth/qwen2.5-3b-instruct-bnb-4bit`
- LoRA rank: 16, alpha: 16
- Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- GRPO 300 steps, batch 2x4=8, fp16
- Google Colab T4 GPU, ~219 minutes

---

## The Dataset

<p align="center">
  <img src="assets/assets/medsentinel_dataset_overview.png" width="100%" alt="Dataset distribution" />
</p>
<p align="center"><i>200 cases · 25 conditions · 10 medical categories · 8 cases per condition (balanced)</i></p>

- **200** synthetic emergency patient cases
- **25** unique emergency conditions
- **30** drugs with full dosage ranges, contraindications, interaction profiles
- **35** ICD-10 codes across cardiac, respiratory, neurological, endocrine, infectious
- **146 of 200 cases** include allergy conflict edge cases
- Generated via `tools/generate_patient_cases_anthropic.py`, scales to 2,000+ cases

---

## The CVL: Why We Built a Completely Separate Pipeline

The Doctor can hallucinate. The Auditor catches rule violations but cannot understand clinical context. A patient with appendicitis who gets nitroglycerin passes the Auditor's allergy check because nitroglycerin is not in that patient's allergy list. Technically not a violation. Clinically completely wrong.

One wrong drug in real medicine causes irreversible damage. So we added a final independent review.

**How it works:**

After all agents run and the reward is computed for RL training, the CVL sends everything to Claude API as a completely separate, independent call. It knows nothing about what the Doctor reasoned. It gets the patient record and the proposed output and asks: is this clinically correct?

```
Doctor -> Auditor -> Reward computed (RL training stops here)
                           |
              +------------------------------+
              |  CVL (Claude API)            |
              |  Completely independent      |
              |  Re-checks drug-dx match     |
              |  Validates dose for patient  |
              |  Catches missed interactions |
              |  Overrides if wrong          |
              |  Explains every change       |
              +------------------------------+
                           |
                   Final verified output
```

**The most important design decision:** CVL does NOT affect the reward signal. The RL model trains on its own raw decisions. CVL is a production safety net only.

Without `ANTHROPIC_API_KEY`: CVL skips silently, returns doctor output with `cvl_fallback: true`.

```python
# Disable CVL during training (speed), enable for production
env = MedSentinelEnv(EnvConfig(
    patient_dataset_path="data/patient_cases.json",
    enable_cvl=True,
))
```

---

## API Keys: What They Control

### Doctor Agent

```
ANTHROPIC_API_KEY set     -> Layer 1: Claude API diagnoses
No key, GPU available     -> Layer 2: Qwen2.5-3B LoRA (our trained model)
No key, no GPU            -> Layer 3: pure Python rule-based script
```

Setting the key on HuggingFace Spaces: Settings → Variables and Secrets → `ANTHROPIC_API_KEY`

### Auditor Agent

No API key. No model. Pure Python always. Fully deterministic.

### CVL

Requires `ANTHROPIC_API_KEY`. If not set, skips silently (pass-through mode). Does not affect reward or training either way.

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

# 4. Start everything
python start.py
# Backend: http://localhost:8000
# UI:      http://localhost:8080
```

**Run training notebook:**
```bash
# Open in Google Colab
training/MedSentinel_GRPO_Training.ipynb
```

---

## File Structure

```
medsentinel/
├── agents/
│   ├── doctor_agent.py                     # Claude API + Qwen2.5-3B LoRA + rule-based Python
│   ├── auditor_agent.py                    # Pure Python rule-based safety checker
│   └── clinical_verification_layer.py     # Claude API separate final safety pipeline
├── env/
│   ├── medsentinel_env.py                  # OpenEnv-compliant gym-style env
│   ├── reward_system.py                    # Deterministic reward function
│   └── schema_drift.py                     # Adversarial attacker (pure Python)
├── server/
│   ├── __init__.py
│   ├── app.py                              # OpenEnv FastAPI via create_app()
│   └── medsentinel_environment.py          # Inherits Environment base class
├── tools/
│   ├── mcp_tools.py                        # 5 MCP clinical tools
│   └── generate_patient_cases_anthropic.py # Dataset generator
├── data/
│   ├── patient_cases.json                  # 200 synthetic cases
│   ├── emergency_drugs.json                # 30 drugs with dosage ranges
│   └── icd10_emergency_conditions.json     # 35 ICD-10 codes
├── training/
│   ├── MedSentinel_GRPO_Training.ipynb     # Colab notebook
│   ├── train_grpo.py                       # GRPO training script
│   └── eval_metrics.py                     # Evaluation script
├── medsentinel_weights_to_share/           # Trained LoRA adapter
│   ├── adapter_config.json                 # LoRA config (rank 16, alpha 16)
│   ├── adapter_model.safetensors           # Trained weights (Git LFS)
│   ├── tokenizer.json                      # Tokenizer
│   ├── tokenizer_config.json               # Tokenizer config
│   └── chat_template.jinja                 # Chat template
├── assets/
│   └── assets/                             # All images for README
│       ├── 1777144765641.png               # Architecture diagram
│       ├── 1777144798265.png               # Full pipeline diagram
│       ├── 1777144749858.png               # Model evolution radar
│       ├── Medsentinel.png                 # medsentinel image
│       ├── 1777144770389.png               # Reward scoring table
│       ├── medsentinel_reward_curve.png    # GRPO training curve
│       ├── medsentinel_reward_components.png # Reward component weights
│       └── medsentinel_dataset_overview.png  # Dataset distribution
├── dashboard/
│   ├── app.py                              # Streamlit demo dashboard
│   └── static/reward_curve.png
├── tests/
│   └── integration_run_one_episode.py      # Integration test
├── ui/                                     # React + TypeScript + Vite frontend
│   ├── src/
│   │   ├── components/sections/            # Hero, Pipeline, InteractiveDemo, etc.
│   │   ├── lib/diagnosisEngine.ts          # Real backend call + mock fallback
│   │   └── pages/Index.tsx
│   ├── package.json
│   └── vite.config.ts
├── HF_BLOG_POST.md                         # Full technical blog post
├── api_server.py                           # FastAPI bridge for React UI
├── app_hf.py                               # HuggingFace Spaces entry point (push this to GitHub)
├── models.py                               # OpenEnv Action/Observation/State types
├── openenv.yaml                            # OpenEnv manifest (spec_version: 1)
├── openenv_client.py                       # EnvClient subclass
├── requirements.txt                        # Full local requirements
├── requirements.hf.txt                     # Slim HF requirements (push this to GitHub)
├── training_summary.json                   # Real training results (v1 + v2)
├── start.py                                # Start backend + UI in one command
├── .gitattributes                          # Git LFS tracking for safetensors
└── LICENSE                                 # MIT
```

---

## The Ceiling: What More Compute Would Do

| Setup | Expected Accuracy |
|---|---|
| Current (160 cases, 300 steps, Colab T4) | ~10% |
| 500 cases, 500 steps, 3B | ~25-35% |
| 2,000 cases, 1,000 steps, 3B | ~40-55% |
| MIMIC-IV data, 2,000 steps, 7B | ~65-75% |
| MIMIC-IV + 3,000 steps + A100 x 20-40 hrs | **85-90%** |

The architecture is proven. The reward function is solid. We ran out of GPU credits, not ideas.

---

## OpenEnv Compliance

| Requirement | Status | File |
|---|---|---|
| `Environment` base class | ✅ | `server/medsentinel_environment.py` |
| `Action` / `Observation` / `State` types | ✅ | `models.py` (openenv-core types) |
| `create_app()` server | ✅ | `server/app.py` |
| `EnvClient` subclass | ✅ | `openenv_client.py` |
| Valid `openenv.yaml` (spec_version: 1) | ✅ | `openenv.yaml` |
| HuggingFace Spaces Docker deployment | ✅ | `Dockerfile` + `app_hf.py` |

---

## Theme Compatibility

| Theme | How MedSentinel fits |
|---|---|
| **Multi-Agent Interactions** | Doctor + Auditor + Attacker each play different roles every episode |
| **World Modeling** | Real tool interactions, stateful MCP calls, multi-step clinical workflow |
| **Self-Improvement** | Reward hacking caught, diagnosed precisely, fixed with 3 targeted changes, retrained |
| **Wild Card** | Schema drift as first-class adversarial training objective, genuinely novel in medical RL |

---

## License

MIT · Built for research and education · Not for clinical use

*OpenEnv Hackathon India 2026 · Team of 3*
