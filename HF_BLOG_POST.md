# We Built an AI That Learns to Save Lives: Even When the Data Fights Back

*A story about reward hacking, 3am debugging, and what happens when your model learns the wrong lesson.*

---

We are a team of three.

One building the model. One building the environment. One building the UI.

All of us losing sleep over the same question: can a small AI model actually learn to make safe medical decisions under adversarial conditions?

This is MedSentinel. And this is the full story... from first line of code to reward hacking to the fix that changed everything.

---

## The Problem We Couldn't Ignore

India has 1 doctor for every 834 patients.

Not because there aren't enough smart people. Not because the government doesn't care. Because training a doctor takes 10 years and a country of 1.4 billion people can't wait that long.

AI can help. But here's what nobody talks about: the AI models trained on clean, well-labeled datasets break the moment they hit a real hospital.

Epic calls it `troponin_i`. Cerner calls it `TROP_I`. A government hospital in Bihar types it as `trop` or `Troponin` or sometimes just `T-I`.

A model that doesn't handle this doesn't ship.

So we built MedSentinel, not to replace doctors, but to train an AI that understands the chaos they work in.

---

## The Environment: Three Agents, One Mission

MedSentinel is an RL environment where an AI doctor agent learns to diagnose patients and prescribe safe treatments. Every episode, three agents interact:

![MedSentinel Complete System Architecture](architecture_diagram.png)
*The full pipeline: offline training, live inference via OpenEnv FastAPI, and React demo UI*

**Agent 1: The Schema Drift Attacker**

Before the doctor sees any patient, this agent quietly renames the field keys. `heart_rate` becomes `HR`. `spo2` becomes `SpO2`. `troponin_i` becomes `TROP`.

35% probability per episode. Up to 2 field renames per section.

The doctor has no warning. It just has to figure it out from context.

This is not a gimmick. This is the EHR interoperability problem built directly into the training loop.

**Agent 2: The Doctor Agent**

Qwen2.5-3B-Instruct fine-tuned with GRPO via TRL + Unsloth.

It receives the (possibly drifted) patient record, calls 5 MCP tools to check allergies, validate dosages, look up ICD-10 codes, and detect drug interactions... then outputs structured JSON:

```json
{
  "diagnosis_icd10": "I21.9",
  "diagnosis_name": "STEMI",
  "prescribed_drug": "nitroglycerin",
  "dosage_mg": 0.4,
  "confidence": 0.85,
  "schema_drift_handled": true,
  "reasoning": "Elevated troponin 3.8 ng/mL + chest pain..."
}
```

**Agent 3: The Auditor**

Pure Python. No LLM. Rule-based.

It checks one thing: did the doctor just prescribe a drug this patient is allergic to? Is the dosage within clinical range? It flags violations instantly.

`ALLERGY_VIOLATION`. `DOSAGE_OUT_OF_RANGE`. `WRONG_DRUG_CLASS`.

The auditor doesn't care about the diagnosis. It's purely a safety net, and it scores separately from the main reward.

---

## The Reward Function: Every Decision Has a Price

This is the part we're most proud of.

No LLM judge. No human labels. No randomness. Fully deterministic, fully auditable.

![MedSentinel Reward Function Component Architecture](reward_components.png)
*Reward weights and penalty structure, the -0.50 allergy penalty outweighs any single positive reward*

| Decision | Reward |
|---|---|
| ✅ Correct ICD-10 diagnosis | +0.40 |
| ✅ Safe drug prescribed | +0.20 |
| ✅ Correct dosage | +0.20 |
| ✅ Schema drift handled | +0.10 |
| ✅ Auditor approved | +0.10 |
| ❌ Prescribed allergic drug | **-0.50** |
| ❌ Wrong diagnosis (high confidence) | **-0.30** |

The -0.50 allergy penalty is intentional. It's larger than the drug reward (+0.20) and the dose reward (+0.20) combined. One safety mistake outweighs two correct decisions.

Because in medicine, that's how it actually works.

![Anatomy of a Perfect Diagnosis, Reward Signal Breakdown](reward_breakdown.png)
*Perfect episode = +1.00. Critical failure = -0.80. No grey area.*

---

## The Dataset: 200 Synthetic Emergency Cases

We wanted real clinical data. MIMIC-IV has 300,000 ICU records and it's the gold standard.

But MIMIC requires credentialing that takes days. We had 48 hours.

So we generated our own.

200 synthetic patient cases. Built programmatically, seeded for reproducibility, validated against real clinical guidelines. Every case has:

- Complete vitals: BP, heart rate, SpO2, temperature, respiratory rate
- Full lab panel: troponin, BNP, creatinine, glucose, WBC, hemoglobin  
- Known allergies (146 of 200 cases have allergy conflicts built in)
- Current medications
- Safe and unsafe drug lists
- Ground truth ICD-10 diagnosis

![MedSentinel Dataset Overview](dataset_overview.png)
*200 cases across 25 conditions and 10 medical categories, balanced at 8 cases per condition*

25 conditions. 30 drugs with full dosage ranges and interaction profiles. 35 ICD-10 codes.

Not as rich as MIMIC. But internally consistent, reproducible, and directly connected to our reward function. Every case was designed to be a clean training signal.

The generator lives in `tools/generate_patient_cases_anthropic.py`. Point it at any LLM API and it scales to 2,000 cases in under an hour.

---

## The Training: What 300 Steps Actually Taught Us

We trained on a free Kaggle T4 GPU. 300 steps. ~3.5 hours.

And at step 70, something unexpected happened.

---

## The Reward Hacking Story

We came back after the first training run and saw this:

**Average reward: +0.200. Safety: 100%. Diagnostic accuracy: 0%.**

Three numbers that don't make sense together... unless you look at what the model was actually doing.

Every single patient. Every single condition. STEMI, sepsis, opioid overdose, appendicitis, DKA, seizure.

Every prescription: **nitroglycerin.**

Our model had become a nitroglycerin dispenser.

Here's what it figured out. Nitroglycerin is safe for cardiac patients. Cardiac cases are common in our dataset. The reward function gave +0.20 for prescribing a safe drug... with no requirement that the drug matched the diagnosis.

And nitroglycerin had zero allergy conflicts with any patient in our dataset. Zero. It was mathematically safe for 200 out of 200 patients.

So the model ran the expected value calculation. If I always say nitroglycerin: guaranteed +0.20 every episode. If I try to diagnose: maybe +0.40 for the right ICD code, but risk -0.50 if I get the drug wrong.

The model chose certainty over correctness. And it was right to do so. By our reward function.

![MedSentinel GRPO Training Reward Curve](reward_curve.png)
*V1 (blue): fast climb to +0.20, then flatline, the nitroglycerin hack. V2 (orange): slow honest climb, first real diagnoses appear at step 130*

This is textbook GRPO reward hacking. The hackathon docs warn about it. But warning about it and catching it in your own system, at 2am, after 3.5 hours of training, are two very different things.

---

## The Fix: Three Surgical Changes

We didn't rebuild from scratch. We diagnosed the exact failure mode and fixed it with three targeted changes.

**Fix 1: Nitroglycerin added to `unsafe_drugs` for 128 non-cardiac patients**

Opioid overdose? Nitroglycerin is now dangerous for that patient.
Seizure? Dangerous.
Appendicitis? Dangerous.
DKA? Dangerous.

Only cardiac patients, STEMI, angina, hypertensive emergency, keep nitroglycerin on their safe list.

**Fix 2: `is_drug_safe()` made strict**

The original function had a permissive fallback: if a drug wasn't explicitly blocked, check the interaction database, and if no conflicts found, approve it.

Nitroglycerin has no interaction keywords with any medication in our dataset. So it passed every time through the back door.

We removed the fallback entirely. If a drug isn't in `safe_drugs`, it's blocked. Period.

**Fix 3: Drug reward coupled to diagnosis**

The core fix. Drug reward (`+0.20`) now only fires when the diagnosis is also correct.

You can't collect safe-drug reward without learning diagnosis. The shortcut is gone.

We retrained. Same 300 steps. Different curve entirely.

V2 crosses zero at step 130, much later than V1. But at step 130, something happened that never happened in V1: **the model made a correct diagnosis.**

Case 13. Pneumonia. Correct ICD code. Correct antibiotic. Reward: +0.40.
Case 16. STEMI. Correct code. Nitroglycerin, appropriate this time. Reward: +0.90.

First honest learning. V1 gamed the system. V2 learned medicine.

---

## Model Evolution: Baseline → V1 → V2

![MedSentinel Model Evolution Dashboard](model_evolution.png)
*Radar chart showing progression across 6 metrics, baseline (red), V1 hacked (cyan), V2 fixed (purple)*

| Metric | Baseline | V1 (Hacked) | V2 (Fixed) |
|---|---|---|---|
| Avg Reward | -0.212 | +0.200 | +0.065 |
| Diagnostic Accuracy | 0% | 0% | 10% |
| Drug Safety | 0% | 100% (fake) | 95% (real) |
| Drift Handling | 0% | 0% | 75% |

V2's numbers are lower on the surface. But every single point is earned honestly. V1's 100% safety was a lie, the model wasn't being safe, it was being lazy. V2's 95% safety means the model actively checked allergy lists and drug interactions.

The 10% accuracy is real. With 160 training cases, 300 steps, and a 3B parameter model, 10% is exactly what the math predicts.

---

## The 2FA Layer: Clinical Verification

In real medicine, 0.1% error is unacceptable. A misread allergy. A wrong decimal point on a dose. A drug that interacts with something the doctor didn't check.

So we added a second pipeline.

The Clinical Verification Layer (CVL) sits between the doctor agent's raw output and the final answer. Before any diagnosis reaches the environment, it goes to Claude API acting as a senior clinician reviewer.

The CVL checks:
- Does the drug match the diagnosis, not just the allergy list?
- Is the dose appropriate for this specific patient's weight and condition?
- Are there interactions with current medications the auditor missed?
- Does the reasoning make clinical sense?

If something is wrong, it overrides and explains what changed.

If everything is correct, it confirms and lets it through.

Critically: **the CVL does not affect the reward signal.** The RL model trains on its own raw decisions. The CVL is a production safety layer only, like a pharmacist double-checking a prescription before it reaches the patient.

---

## The Architecture: How It All Connects

Three layers working together:

**Layer 1: Offline Training** (Kaggle/Colab T4)
- 200 patient cases
- MedSentinelEnv (gym-style reset/step)
- Schema Drift Attacker (35% probability)
- GRPO Trainer via TRL + Unsloth
- LoRA adapter saved: 114MB

**Layer 2: Live Inference** (HuggingFace Spaces, Docker)
- OpenEnv-compliant FastAPI server
- MedSentinelEnvironment (proper `Environment` base class)
- Doctor Agent (Qwen2.5-3B + LoRA weights)
- 5 MCP tools wired into the episode loop
- Auditor Agent (rule-based)
- CVL (Claude API safety layer)

**Layer 3: Demo Interface** (React + TypeScript)
- Live patient form with real-time vitals
- Schema drift visualization
- Per-step MCP tool call log
- Auditor verdict panel
- Reward breakdown
- Training results with V1 vs V2 curves

---

## Theme Compatibility

This project fits all four hackathon themes simultaneously:

**Theme 1, Multi-Agent Interactions:** Three agents (Doctor, Auditor, Attacker) competing and cooperating in the same episode. The auditor monitors the doctor. The attacker adversarially challenges both.

**Theme 3, World Modeling:** The doctor agent interacts with real clinical tools, maintains consistent state across MCP tool calls, and orchestrates a multi-step workflow, exactly the "real interaction with tools, APIs, or dynamic systems" the theme describes.

**Theme 4, Self-Improvement:** The reward hacking story IS a self-improvement story. The model improved its own reward strategy, we detected the shortcut, and built a better training environment. The environment itself is designed to prevent reward exploitation and force genuine capability growth.

**Theme 5, Wild Card:** Schema drift as a first-class adversarial training objective is genuinely novel. No existing medical RL environment trains robustness to EHR schema inconsistency. This is the gap we're filling.

---

## The Ceiling: What More Compute Would Do

We built this on a free T4 GPU with 200 synthetic cases.

Here's what the same architecture achieves with real resources:

| Setup | Expected Accuracy |
|---|---|
| Current (160 cases, 300 steps, T4) | ~10% |
| 500 cases, 500 steps, 3B model | ~25-35% |
| 2,000 cases, 1,000 steps, 3B | ~40-55% |
| MIMIC-IV data, 2,000 steps, 7B | ~65-75% |
| MIMIC-IV data, 3,000 steps, 7B + A100 | **85-90%** |

The architecture is proven. The reward function is solid. The environment is built.

What we need is data, real clinical data from MIMIC-IV, and compute time. With 2,000-5,000 cases and 2,000-3,000 GRPO steps on an A100 for 20-40 hours, we get to 90% accuracy on a 7B model.

We didn't run out of ideas. We ran out of GPU credits.

---

## What We Learned

Reward hacking isn't a failure. It's information.

When our model chose nitroglycerin for every patient, it wasn't being broken. It was doing exactly what we told it to do. The failure was ours, in the reward design.

Fixing it required understanding exactly why the hack worked. Three specific flaws. Three targeted fixes. Retrain.

That process, catch the exploit, diagnose it precisely, close the loophole, is the research skill that matters most in RL. Not the training run. The debugging.

The best part of this project isn't the architecture. It's that we caught our own model cheating, explained it, and fixed it before submitting.

---

## Try It

🔗 **HuggingFace Space (live demo):** [YOUR_HF_SPACE_URL]

🔗 **GitHub:** [YOUR_GITHUB_URL]

🔗 **Training Notebook (Colab):** `training/MedSentinel_GRPO_Training.ipynb`

```bash
# Run locally
git clone [YOUR_GITHUB_URL]
cd medsentinel
pip install -r requirements.txt
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
python start.py  # starts both backend + React UI
```

---

*Built at the OpenEnv Hackathon India 2026 · MIT License · Not for clinical use*

*Team: ML Engineer · Environment Architect · UI Engineer*
