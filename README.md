---
title: MedSentinel
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
app_file: server/app.py
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
| 🚀 Live Demo (HuggingFace Spaces) | [YOUR_HF_SPACE_URL] |
| 📓 Training Notebook (Colab) | [Open in Colab](training/MedSentinel_GRPO_Training.ipynb) |
| 📝 Full Blog Post | [HuggingFace Blog](YOUR_HF_BLOG_URL) |
| 💻 GitHub Repository | [YOUR_GITHUB_URL] |
| 🤗 Trained Model Weights | [YOUR_HF_MODEL_URL] |

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
💀 Schema Drift Attacker     ← renames vitals/lab field keys at 35% probability
       ↓                        troponin_i → TROP | heart_rate → HR | spo2 → SpO2
🩺 Doctor Agent (Qwen2.5-3B) ← calls 5 MCP tools, outputs ICD-10 + drug + dose
       ↓
🛡️ Auditor Agent (rule-based) ← flags ALLERGY_VIOLATION, DOSAGE_OUT_OF_RANGE
       ↓
🏆 Deterministic Reward       ← no LLM judge, fully auditable
       ↓
🔐 CVL (Claude API)           ← 2FA safety layer, not in reward loop
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

## Clinical Verification Layer (2FA)

In medicine, 0.1% error has consequences.

The CVL sits between the doctor agent's output and the final answer. Before any decision is returned, Claude API acts as a senior clinician reviewer, checking drug-diagnosis match, dose appropriateness, and missed interactions.

**Key design:** CVL does NOT affect the reward signal. The RL model trains on its own raw decisions. CVL is a production safety net only.

```python
# Enable CVL for production (disable during training for speed)
env = MedSentinelEnv(EnvConfig(
    patient_dataset_path="data/patient_cases.json",
    enable_cvl=True,  # False during GRPO training
))
```

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
