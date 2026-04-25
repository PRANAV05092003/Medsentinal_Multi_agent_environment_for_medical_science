#!/usr/bin/env python3
"""
MedSentinel Dashboard — Full Demo UI
=====================================
Features:
- Dataset mode + Custom JSON mode
- Schema drift visualization
- Doctor agent output (Anthropic or local rule-based)
- MCP tool call log (per episode)
- Auditor flags panel
- Reward breakdown
- Live reward curve (this session)
- Training reward curve (from saved PNG or training_summary.json)
- Before/After comparison (baseline vs trained)
- Episode history table + JSON export
- Mobile-friendly layout

Run:
  streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from typing import Any, Dict, List, Optional

import streamlit as st

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from agents.doctor_agent import DoctorAgent
from agents.auditor_agent import audit_doctor_output
from env.medsentinel_env import EnvConfig, MedSentinelEnv
from env.reward_system import compute_reward
from env.schema_drift import apply_schema_drift
from tools.mcp_tools import check_allergies, dose_check, icd_lookup, query_labs, drug_interactions

DEFAULT_DATASET = os.path.join(REPO_ROOT, "data", "patient_cases.json")
TRAINING_SUMMARY_PATH = os.path.join(REPO_ROOT, "checkpoints", "medsentinel_grpo", "training_summary.json")
REWARD_CURVE_PNG = os.path.join(REPO_ROOT, "dashboard", "static", "reward_curve.png")

CUSTOM_TEMPLATE: Dict[str, Any] = {
    "patient_id": "P-CUSTOM-001",
    "age": 58, "gender": "male",
    "chief_complaint": "Crushing chest pain for 45 minutes with diaphoresis.",
    "vitals": {"bp_systolic": 154, "bp_diastolic": 90, "heart_rate": 108,
               "temperature": 37.1, "spo2": 95, "respiratory_rate": 22},
    "lab_results": {"troponin_i": 2.8, "bnp": 180, "creatinine": 1.0,
                    "glucose": 132, "wbc": 9.2, "hemoglobin": 14.0},
    "known_allergies": ["aspirin"],
    "current_medications": ["lisinopril"],
    "ground_truth_diagnosis": "I21.9",
    "safe_drugs": ["nitroglycerin"],
    "unsafe_drugs": ["aspirin"],
}


# ─── Helpers ───────────────────────────────────────────────────────────────

def _fallback_dataset():
    return [{
        "patient_id": "PT-DEMO-1", "age": 58, "gender": "male",
        "chief_complaint": "severe chest pain and diaphoresis",
        "vitals": {"bp_systolic": 92, "bp_diastolic": 58, "heart_rate": 118,
                   "temperature": 36.8, "spo2": 94, "respiratory_rate": 22},
        "lab_results": {"troponin_i": 0.44, "bnp": 120, "creatinine": 1.2,
                        "glucose": 115, "wbc": 9.6, "hemoglobin": 14.2},
        "known_allergies": ["aspirin"],
        "current_medications": [],
        "ground_truth_diagnosis": "I21.9",
        "safe_drugs": ["nitroglycerin"],
        "unsafe_drugs": ["aspirin"],
    }]


def _write_temp(cases):
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    with open(path, "w") as f:
        json.dump(cases, f)
    return path


def _pjson(obj):
    try:
        return json.dumps(obj, indent=2, ensure_ascii=False)
    except Exception:
        return str(obj)


def _run_mcp_tools(patient: Dict[str, Any], drug: str, dose: Optional[float]) -> List[Dict[str, Any]]:
    """Run MCP tools and return a log of calls + results."""
    log = []
    # 1. query_labs
    result = query_labs(patient)
    log.append({"tool": "query_labs", "input": {"patient_id": patient.get("patient_id")}, "result": result})
    # 2. check_allergies
    if drug:
        result = check_allergies(patient, drug)
        log.append({"tool": "check_allergies", "input": {"drug_name": drug}, "result": result})
    # 3. dose_check
    if drug and dose is not None:
        result = dose_check(drug, dose)
        log.append({"tool": "dose_check", "input": {"drug_name": drug, "dose_mg": dose}, "result": result})
    # 4. drug_interactions
    current_meds = patient.get("current_medications", [])
    if drug and current_meds:
        result = drug_interactions(drug, current_meds)
        log.append({"tool": "drug_interactions", "input": {"drug": drug, "meds": current_meds}, "result": result})
    # 5. icd_lookup
    result = icd_lookup(patient.get("ground_truth_diagnosis", ""))
    log.append({"tool": "icd_lookup", "input": {"code": patient.get("ground_truth_diagnosis")}, "result": result})
    return log


def _run_episode(patient_input, doctor, seed, drift_prob, max_renames):
    """Run full pipeline on a patient dict. Returns result dict."""
    patient_original = dict(patient_input)
    patient_obs, drift_occurred, drift_changes = apply_schema_drift(
        patient_original, seed=seed, drift_probability=drift_prob,
        max_key_renames_per_section=max_renames)
    doctor_output = doctor.diagnose(patient_obs)
    drug = doctor_output.get("prescribed_drug", "")
    dose = doctor_output.get("dosage_mg")
    mcp_log = _run_mcp_tools(patient_obs, drug, dose)
    auditor = audit_doctor_output(doctor_output, patient_obs)
    auditor_flags = {"is_correct": bool(auditor.get("safe", False)),
                     "flags": auditor.get("flags", [])}
    reward, breakdown = compute_reward(doctor_output, patient_obs,
                                       auditor_flags=auditor_flags,
                                       drift_flag=bool(drift_occurred))
    return {
        "patient_original": patient_original,
        "patient_observed": patient_obs,
        "drift_occurred": bool(drift_occurred),
        "drift_changes": drift_changes,
        "doctor_output": doctor_output,
        "mcp_log": mcp_log,
        "auditor": auditor,
        "reward": float(reward),
        "reward_breakdown": breakdown,
    }


def _init_state():
    st.session_state.setdefault("last_run", None)
    st.session_state.setdefault("temp_path", None)
    st.session_state.setdefault("custom_json", _pjson(CUSTOM_TEMPLATE))
    st.session_state.setdefault("reward_history", [])
    st.session_state.setdefault("episode_log", [])


# ─── Rendering components ──────────────────────────────────────────────────

def render_reward_color(reward: float) -> str:
    if reward >= 0.6:
        return "🟢"
    elif reward >= 0.2:
        return "🟡"
    else:
        return "🔴"


def render_patient_panel(patient: Dict, title: str):
    st.markdown(f"**{title}**")
    # Format as clean EHR view
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Demographics**")
        st.write(f"ID: `{patient.get('patient_id', 'N/A')}`")
        st.write(f"Age: {patient.get('age', '?')} | Gender: {patient.get('gender', '?')}")
        st.write(f"**Chief Complaint:** {patient.get('chief_complaint', '')}")
        st.markdown("**Allergies**")
        allergies = patient.get("known_allergies", [])
        if allergies:
            st.error(f"⚠️ {', '.join(allergies)}")
        else:
            st.success("None known")
        st.markdown("**Current Medications**")
        meds = patient.get("current_medications", [])
        st.write(", ".join(meds) if meds else "None")
    with col2:
        st.markdown("**Vitals**")
        vitals = patient.get("vitals", {})
        for k, v in vitals.items():
            st.write(f"`{k}`: {v}")
        st.markdown("**Labs**")
        labs = patient.get("lab_results", {})
        for k, v in labs.items():
            st.write(f"`{k}`: {v}")


def render_drift_panel(drift_occurred: bool, drift_changes: Dict):
    if drift_occurred:
        vitals_drift = drift_changes.get("vitals", {})
        labs_drift = drift_changes.get("lab_results", {})
        all_changes = []
        for old, new in vitals_drift.items():
            all_changes.append({"section": "vitals", "original_key": old, "renamed_to": new})
        for old, new in labs_drift.items():
            all_changes.append({"section": "lab_results", "original_key": old, "renamed_to": new})
        if all_changes:
            st.warning("🔀 Schema drift active — keys renamed by attacker")
            import pandas as pd
            st.dataframe(pd.DataFrame(all_changes), use_container_width=True, hide_index=True)
        else:
            st.warning("🔀 Schema drift triggered but no eligible keys found")
    else:
        st.success("✅ No schema drift in this episode")


def render_mcp_tools_panel(mcp_log: List[Dict]):
    st.markdown("**MCP Tool Calls**")
    for entry in mcp_log:
        tool = entry["tool"]
        result = entry["result"]
        verdict = result.get("verdict", result.get("drift_detected", ""))
        label = f"🔧 `{tool}`"
        if "verdict" in result:
            color = "🟢" if result["verdict"] == "safe" else "🔴" if result["verdict"] == "unsafe" else "🟡"
            label += f" → {color} {result['verdict']}"
        elif "drift_detected" in result:
            label += f" → {'⚠️ drift' if result['drift_detected'] else '✅ no drift'}"
        with st.expander(label, expanded=False):
            st.json({"input": entry["input"], "result": result}, expanded=False)


def render_doctor_panel(doctor_output: Dict):
    st.markdown("**Doctor Agent Output**")
    diag = doctor_output.get("diagnosis_icd10", "N/A")
    diag_name = doctor_output.get("diagnosis_name", "")
    drug = doctor_output.get("prescribed_drug", "N/A")
    dose = doctor_output.get("dosage_mg", "N/A")
    conf = doctor_output.get("confidence", 0)
    drift_handled = doctor_output.get("schema_drift_handled", False)

    c1, c2, c3 = st.columns(3)
    c1.metric("Diagnosis", diag)
    c2.metric("Drug", drug)
    c3.metric("Dose (mg)", dose if dose is not None else "null")
    st.progress(float(conf) if conf else 0, text=f"Confidence: {conf:.0%}")
    if drift_handled:
        st.info("✅ Agent reported schema drift handled")
    else:
        st.write("Schema drift handled: False")
    with st.expander("Full reasoning", expanded=False):
        st.write(doctor_output.get("reasoning", "No reasoning provided."))
    with st.expander("Raw JSON", expanded=False):
        st.json(doctor_output)


def render_auditor_panel(auditor: Dict):
    st.markdown("**Auditor Verdict**")
    safe = auditor.get("safe", False)
    if safe:
        st.success("✅ SAFE — No violations found")
    else:
        st.error("❌ UNSAFE — Violations detected")
    flags = auditor.get("flags", [])
    if flags:
        for flag in flags:
            st.warning(f"🚩 `{flag}`")
    notes = auditor.get("notes", [])
    if notes:
        with st.expander("Auditor notes", expanded=False):
            for note in notes:
                st.write(f"• {note}")


def render_reward_panel(reward: float, breakdown: Dict):
    st.markdown("**Reward Breakdown**")
    icon = render_reward_color(reward)
    st.metric(f"{icon} Episode Reward", f"{reward:.3f}", help="Max possible: 1.0 | Min: -0.8")
    components = breakdown.get("components", {})
    penalties = breakdown.get("penalties", {})
    rows = []
    for k, v in components.items():
        rows.append({"component": k, "value": v, "type": "reward"})
    for k, v in penalties.items():
        rows.append({"component": k, "value": v, "type": "penalty"})
    if rows:
        import pandas as pd
        df = pd.DataFrame(rows)
        df["value"] = df["value"].apply(lambda x: f"{x:+.2f}")
        st.dataframe(df, use_container_width=True, hide_index=True)


def render_live_curve():
    history = st.session_state["reward_history"]
    if not history:
        return
    st.markdown("**Live Session Reward Curve**")
    import pandas as pd
    df = pd.DataFrame({"Episode": list(range(1, len(history)+1)), "Reward": history})
    st.line_chart(df, x="Episode", y="Reward", height=200)
    c1, c2, c3 = st.columns(3)
    c1.metric("Episodes", len(history))
    c2.metric("Avg Reward", f"{sum(history)/len(history):.3f}")
    c3.metric("Best Reward", f"{max(history):.3f}")


def render_training_curve():
    """Show training reward curve from saved training run."""
    st.markdown("**GRPO Training Reward Curve**")
    if os.path.exists(TRAINING_SUMMARY_PATH):
        with open(TRAINING_SUMMARY_PATH) as f:
            summary = json.load(f)
        reward_log = summary.get("reward_log", [])
        if reward_log:
            import pandas as pd
            df = pd.DataFrame({"Step": list(range(1, len(reward_log)+1)), "Reward": reward_log})
            st.line_chart(df, x="Step", y="Reward", height=200)
            baseline = summary.get("baseline", {})
            post = summary.get("post_training", {})
            c1, c2, c3 = st.columns(3)
            c1.metric("Steps trained", summary.get("training_steps", "N/A"))
            c2.metric("Baseline accuracy", f"{baseline.get('accuracy_pct', 0):.1f}%")
            c3.metric("Post-train accuracy", f"{post.get('accuracy_pct', 0):.1f}%",
                      delta=f"{post.get('accuracy_pct',0)-baseline.get('accuracy_pct',0):+.1f}%")
        else:
            st.info("Training summary found but reward_log is empty.")
    elif os.path.exists(REWARD_CURVE_PNG):
        st.image(REWARD_CURVE_PNG, caption="Training reward curve (saved image)")
    else:
        st.info("No training results yet. Run `python training/train_grpo.py` to generate.")


def render_before_after():
    """Before/After comparison panel from training_summary.json."""
    st.markdown("**Before vs After Training**")
    if not os.path.exists(TRAINING_SUMMARY_PATH):
        st.info("Run training first to see before/after comparison.")
        return
    with open(TRAINING_SUMMARY_PATH) as f:
        summary = json.load(f)
    baseline = summary.get("baseline", {})
    post = summary.get("post_training", {})
    import pandas as pd
    comparison = {
        "Metric": ["Avg Reward", "Accuracy %", "Safety %"],
        "Before Training": [
            f"{baseline.get('avg_reward', 0):.3f}",
            f"{baseline.get('accuracy_pct', 0):.1f}%",
            f"{baseline.get('safety_pct', 0):.1f}%",
        ],
        "After Training": [
            f"{post.get('avg_reward', 0):.3f}",
            f"{post.get('accuracy_pct', 0):.1f}%",
            f"{post.get('safety_pct', 0):.1f}%",
        ],
    }
    st.dataframe(pd.DataFrame(comparison), use_container_width=True, hide_index=True)
    st.caption(f"Model: {summary.get('base_model', 'N/A')} | "
               f"Steps: {summary.get('training_steps', 'N/A')} | "
               f"Runtime: {summary.get('elapsed_minutes', 0):.1f} min")


# ─── Main ──────────────────────────────────────────────────────────────────


def render_cvl_panel(cvl_output: dict):
    """Render the Clinical Verification Layer panel."""
    st.markdown("**Clinical Verification Layer (2FA)**")

    if not cvl_output:
        st.info("ℹ️ CVL not active — add ANTHROPIC_API_KEY to enable")
        return

    if cvl_output.get("cvl_fallback"):
        st.warning(f"⚠️ CVL in fallback mode: {cvl_output.get('cvl_notes','')}")
    elif cvl_output.get("cvl_verified"):
        changes = cvl_output.get("cvl_changes", [])
        if changes:
            st.warning(f"🔄 CVL made {len(changes)} correction(s)")
            for ch in changes:
                st.write(f"  • {ch}")
        else:
            st.success("✅ CVL confirmed — doctor output verified, no changes needed")

    risk_flags = cvl_output.get("cvl_risk_flags", [])
    if risk_flags:
        st.error("🚨 Residual risk flags after CVL:")
        for flag in risk_flags:
            st.write(f"  • {flag}")

    # Show what CVL decided
    col1, col2 = st.columns(2)
    col1.metric("Verified Diagnosis", cvl_output.get("diagnosis_icd10", "N/A"))
    col2.metric("Verified Drug", cvl_output.get("prescribed_drug", "N/A"))

    with st.expander("CVL Full Reasoning", expanded=False):
        st.write(cvl_output.get("reasoning", "No reasoning provided."))
    with st.expander("CVL Notes", expanded=False):
        st.write(cvl_output.get("cvl_notes", "None"))


def main():
    st.set_page_config(
        page_title="MedSentinel",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _init_state()

    # Custom CSS for mobile-friendliness
    st.markdown("""
    <style>
    .block-container { padding-top: 1rem; padding-bottom: 1rem; }
    @media (max-width: 768px) {
        .stColumns { flex-direction: column; }
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("🏥 MedSentinel")
    st.caption("Multi-agent medical RL — schema drift attacker → doctor agent → auditor → deterministic reward")

    # ── Sidebar ──────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Configuration")

        input_source = st.radio("Input source", ["Dataset", "Custom JSON"], horizontal=True)
        dataset_path = st.text_input("Dataset path", DEFAULT_DATASET,
                                     disabled=(input_source != "Dataset"))
        mode = st.selectbox("Split", ["train", "test"], index=1)
        seed = int(st.number_input("Seed", value=123, min_value=0))
        drift_prob = st.slider("Drift probability", 0.0, 1.0, 0.75, 0.05)
        max_renames = st.slider("Max key renames", 0, 5, 2)

        st.divider()
        provider = st.selectbox("Doctor provider", ["anthropic", "local"])
        has_key = bool(os.environ.get("ANTHROPIC_API_KEY") or
                       os.path.exists(os.path.join(REPO_ROOT, ".env")))
        if provider == "anthropic" and not has_key:
            st.warning("No API key — will use local provider")
        anthropic_model = st.text_input("Anthropic model", os.environ.get("ANTHROPIC_MODEL", ""))

        st.divider()
        run_btn = st.button("▶ Run Episode", type="primary", use_container_width=True)

        st.divider()
        st.markdown("**System health**")
        st.write("Drug DB:", "✅" if os.path.exists(os.path.join(REPO_ROOT, "data", "emergency_drugs.json")) else "❌")
        st.write("ICD DB:", "✅" if os.path.exists(os.path.join(REPO_ROOT, "data", "icd10_emergency_conditions.json")) else "❌")
        st.write("Dataset:", "✅" if os.path.exists(DEFAULT_DATASET) else "⚠️ fallback")
        st.write("Model:", "✅" if os.path.exists(os.path.join(REPO_ROOT, "checkpoints")) else "🔲 not trained")

    # ── Custom JSON input ────────────────────────────────────
    if input_source == "Custom JSON":
        with st.expander("📋 Custom Patient JSON", expanded=True):
            custom_json = st.text_area("Patient record", st.session_state["custom_json"], height=300)
            st.session_state["custom_json"] = custom_json

    # ── Resolve dataset ──────────────────────────────────────
    resolved_path = dataset_path
    if input_source == "Dataset" and not os.path.exists(resolved_path):
        if not st.session_state["temp_path"]:
            st.session_state["temp_path"] = _write_temp(_fallback_dataset())
        resolved_path = st.session_state["temp_path"]
        st.warning("Dataset not found — using built-in demo case")

    # ── Run episode ──────────────────────────────────────────
    if run_btn:
        with st.spinner("Running episode..."):
            try:
                # Build doctor
                actual_provider = provider
                if provider == "anthropic" and not has_key:
                    actual_provider = "local"
                try:
                    doctor = DoctorAgent(provider=actual_provider,
                                         anthropic_model=anthropic_model or None,
                                         seed=seed)
                except Exception:
                    doctor = DoctorAgent(provider="local", seed=seed)

                if input_source == "Dataset":
                    env = MedSentinelEnv(EnvConfig(
                        patient_dataset_path=resolved_path, mode=mode,
                        seed=seed, drift_probability=drift_prob,
                        max_key_renames_per_section=max_renames))
                    patient_obs = env.reset()
                    patient_original = env.current_patient_original or {}
                    drift_occurred = env.current_drift_info["drift_occurred"]
                    drift_changes = env.current_drift_info["drift_changes"]
                    doctor_output = doctor.diagnose(patient_obs)
                    drug = doctor_output.get("prescribed_drug", "")
                    dose = doctor_output.get("dosage_mg")
                    mcp_log = _run_mcp_tools(patient_obs, drug, dose)
                    _, _, info = env.step(doctor_output)
                    auditor = info["auditor"]
                    reward = info["reward_breakdown"]["reward"]
                    breakdown = info["reward_breakdown"]
                    result = {
                        "patient_original": patient_original,
                        "patient_observed": patient_obs,
                        "drift_occurred": drift_occurred,
                        "drift_changes": drift_changes,
                        "doctor_output": doctor_output,
                        "mcp_log": mcp_log,
                        "auditor": auditor,
                        "reward": float(reward),
                        "reward_breakdown": breakdown,
                    }
                else:
                    patient_input = json.loads(st.session_state["custom_json"])
                    result = _run_episode(patient_input, doctor, seed, drift_prob, max_renames)

                st.session_state["last_run"] = result
                st.session_state["reward_history"].append(result["reward"])
                st.session_state["episode_log"].append({
                    "episode": len(st.session_state["reward_history"]),
                    "patient_id": result["patient_original"].get("patient_id", "?"),
                    "diagnosis": result["doctor_output"].get("diagnosis_icd10", ""),
                    "drug": result["doctor_output"].get("prescribed_drug", ""),
                    "dose_mg": result["doctor_output"].get("dosage_mg", ""),
                    "reward": round(result["reward"], 3),
                    "safe": result["auditor"].get("safe", False),
                    "drift": result["drift_occurred"],
                })
                st.success(f"Episode complete — Reward: {result['reward']:.3f}")
            except Exception as e:
                st.error(f"Episode failed: {e}")
                import traceback
                with st.expander("Traceback"):
                    st.code(traceback.format_exc())

    # ── Results ──────────────────────────────────────────────
    last = st.session_state.get("last_run")
    if not last:
        st.info("👆 Click **Run Episode** to start")
        st.divider()
        render_training_curve()
        render_before_after()
        return

    # Row 1 — Patient comparison
    st.subheader("Patient Record")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        render_patient_panel(last["patient_original"], "📄 Original")
    with c2:
        render_patient_panel(last["patient_observed"], "🔀 Observed (after drift)")

    # Drift highlight
    render_drift_panel(last["drift_occurred"], last["drift_changes"])

    st.divider()

    # Row 2 — Doctor + MCP tools
    c1, c2 = st.columns([3, 2], gap="large")
    with c1:
        st.subheader("🩺 Doctor Agent")
        render_doctor_panel(last["doctor_output"])
    with c2:
        st.subheader("🔧 MCP Tool Calls")
        render_mcp_tools_panel(last["mcp_log"])

    st.divider()

    # Row 3 — Auditor + Reward
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.subheader("🛡️ Auditor")
        render_auditor_panel(last["auditor"])
    with c2:
        st.subheader("🏆 Reward")
        render_reward_panel(last["reward"], last["reward_breakdown"])

    st.divider()

    # Row 3B — Clinical Verification Layer (2FA)
    st.subheader("🔐 Clinical Verification Layer — 2FA Safety Check")
    st.caption("Senior clinician (Claude API) reviews the doctor agent's decision independently of the reward system")
    cvl_out = last.get("cvl_output") if last else None
    render_cvl_panel(cvl_out or {})

    st.divider()

    # Row 4 — Live curve + episode log
    c1, c2 = st.columns(2, gap="large")
    with c1:
        render_live_curve()
    with c2:
        if st.session_state["episode_log"]:
            st.markdown("**Episode History**")
            import pandas as pd
            df = pd.DataFrame(st.session_state["episode_log"])
            st.dataframe(df, use_container_width=True, hide_index=True, height=200)
            st.download_button("💾 Export log (JSON)",
                               data=json.dumps(st.session_state["episode_log"], indent=2),
                               file_name="medsentinel_log.json",
                               mime="application/json")

    st.divider()

    # Row 5 — Training curve + Before/After
    st.subheader("📈 Training Results")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        render_training_curve()
    with c2:
        render_before_after()


if __name__ == "__main__":
    main()
