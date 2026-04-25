---
title: MedSentinel
emoji: 🏥
colorFrom: blue
colorTo: red
sdk: docker
app_file: server/app.py
pinned: false
license: mit
tags:
  - reinforcement-learning
  - medical-ai
  - multi-agent
  - grpo
  - openenv
  - healthcare
---

# MedSentinel 🏥

**Multi-agent Reinforcement Learning for robust medical decision-making**

> A meta-pytorch/OpenEnv compliant environment where a doctor agent learns to diagnose patients under adversarial schema drift attacks.

## Links

- 📓 **Training Notebook (Colab/Kaggle)**: `training/MedSentinel_GRPO_Training.ipynb`
- 📝 **HuggingFace Blog Post**: `HF_BLOG_POST.md`
- 🎤 **Pitch Q&A**: `PITCH_QA.md`
- 🏥 **Live Demo**: HuggingFace Spaces (deploy via `openenv push`)

## Architecture

```
Patient Case (JSON)
       ↓
Schema Drift Attacker         ← renames vitals/lab keys mid-episode
       ↓
Doctor Agent (Qwen2.5-3B)    ← outputs diagnosis + drug + dose as JSON
       ↓
MCP Tools (5 tools)          ← check_allergies, dose_check, icd_lookup, ...
       ↓
Auditor Agent (rule-based)   ← flags ALLERGY_VIOLATION, DOSAGE_OUT_OF_RANGE, ...
       ↓
Deterministic Reward         ← +0.4 diagnosis, +0.2 drug, +0.2 dose, -0.5 allergy
```

## Reward Function

| Signal | Reward |
|---|---|
| ✅ Correct ICD-10 diagnosis | +0.40 |
| ✅ Safe drug prescribed | +0.20 |
| ✅ Correct dosage | +0.20 |
| ✅ Schema drift handled | +0.10 |
| ✅ Auditor approved | +0.10 |
| ❌ Allergic drug | -0.50 |
| ❌ Wrong diagnosis (confident) | -0.30 |

**Max: +1.0 | Min: -0.80 | Fully deterministic — no LLM judge**

## Training Results

| Model | Accuracy | Avg Reward |
|---|---|---|
| Random baseline | ~15% | ~0.0 |
| Zero-shot Qwen2.5-3B | ~35% | ~0.15 |
| GRPO fine-tuned 3B | ~70%+ | ~0.55+ |

## Dataset

200 synthetic emergency patient cases, 25 conditions, 30 drugs, 35 ICD-10 codes.  
146 cases include allergy edge cases.

## MCP Tools

```python
query_labs(patient)              # normalize labs, detect schema drift
check_allergies(patient, drug)   # allergy + unsafe_drugs check
drug_interactions(drug, meds)    # medication interaction check
icd_lookup(code_or_name)         # ICD-10 lookup
dose_check(drug, dose_mg)        # dosage range validation
```

## OpenEnv Integration

This environment uses `openenv-core` properly:
- `server/medsentinel_environment.py` inherits from `openenv.core.env_server.interfaces.Environment`
- `models.py` uses `openenv.core.env_server.types.Action` / `Observation` / `State`
- `server/app.py` uses `openenv.core.env_server.http_server.create_app()`
- `openenv_client.py` inherits from `openenv.core.EnvClient`
- `openenv.yaml` is a valid manifest with `spec_version: 1`

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Add Anthropic key (for demo doctor agent)
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env

# Run OpenEnv server
uvicorn server.app:app --port 7860

# Run dashboard
streamlit run dashboard/app.py

# Run training (Kaggle/Colab — needs GPU)
jupyter nbconvert --to notebook --execute training/MedSentinel_GRPO_Training.ipynb
```

## File Structure

```
medsentinel/
├── server/
│   ├── app.py                      # OpenEnv FastAPI server (create_app)
│   └── medsentinel_environment.py  # Environment(Environment base class)
├── models.py                       # Action / Observation / State (openenv-core types)
├── openenv_client.py               # EnvClient subclass
├── openenv.yaml                    # OpenEnv manifest
├── agents/
│   ├── doctor_agent.py             # LLM + rule-based doctor
│   └── auditor_agent.py            # Rule-based safety auditor
├── env/
│   ├── medsentinel_env.py          # Gym-style env (internal)
│   ├── reward_system.py            # Deterministic reward function
│   └── schema_drift.py             # Schema drift attacker
├── data/
│   ├── patient_cases.json          # 200 synthetic cases
│   ├── emergency_drugs.json        # 30 drugs with dosage ranges
│   └── icd10_emergency_conditions.json  # 35 ICD-10 codes
├── training/
│   ├── train_grpo.py               # GRPO training script
│   ├── MedSentinel_GRPO_Training.ipynb  # ← Colab/Kaggle notebook
│   └── eval_metrics.py             # Evaluation script
├── tools/
│   └── mcp_tools.py                # 5 MCP clinical tools
├── dashboard/
│   └── app.py                      # Streamlit demo dashboard
└── medsentinel_weights_to_share/   # Trained LoRA adapter
    ├── adapter_config.json
    └── adapter_model.safetensors
```

## License

MIT — For research/education only. Not for clinical use.
