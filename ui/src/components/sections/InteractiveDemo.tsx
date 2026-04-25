import { useState, KeyboardEvent } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Stethoscope, ChevronDown, Heart, FlaskConical, ClipboardList, Settings, User, X, Loader2, FileText, Pencil, Upload } from "lucide-react";
import { runDiagnosis, checkBackendHealth, PatientForm, DiagnosisResult } from "@/lib/diagnosisEngine";
import { ResultsPanel } from "./ResultsPanel";
import { PatientUpload } from "./PatientUpload";

const initialForm: PatientForm = {
  patientId: "",
  age: 58,
  gender: "Male",
  chiefComplaint: "Crushing chest pain for 45 minutes with diaphoresis, radiating to left arm.",
  vitals: { bp_systolic: 154, bp_diastolic: 90, heart_rate: 108, temperature: 37.1, spo2: 95, respiratory_rate: 22 },
  labs: { troponin_i: 2.8, bnp: 180, creatinine: 1.0, glucose: 132, wbc: 9.2, hemoglobin: 14.0 },
  allergies: ["aspirin"],
  medications: ["lisinopril"],
  driftEnabled: true,
  driftProbability: 75,
  seed: 123,
};

const VITAL_RANGES = {
  bp_systolic: [90, 140], bp_diastolic: [60, 90], heart_rate: [60, 100],
  temperature: [36.1, 37.5], spo2: [94, 100], respiratory_rate: [12, 20],
};

export const InteractiveDemo = () => {
  const [form, setForm] = useState<PatientForm>(initialForm);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<DiagnosisResult | null>(null);
  const [backendOnline, setBackendOnline] = useState<boolean | null>(null);

  // Check backend health on mount
  useState(() => {
    checkBackendHealth().then(setBackendOnline);
  });
  const [open, setOpen] = useState({ A: true, B: true, C: true, D: true, E: true });
  const [mode, setMode] = useState<"manual" | "upload">("manual");
  const [loadedMeta, setLoadedMeta] = useState<{ filename: string; patientId: string } | null>(null);

  const update = <K extends keyof PatientForm>(k: K, v: PatientForm[K]) => setForm({ ...form, [k]: v });
  const updateNested = (group: "vitals" | "labs", key: string, val: number) =>
    setForm({ ...form, [group]: { ...form[group], [key]: val } });

  const handleLoaded = (partial: Partial<PatientForm>, meta: { filename: string; patientId: string }) => {
    setForm((prev) => ({
      ...prev,
      ...partial,
      vitals: { ...prev.vitals, ...(partial.vitals || {}) },
      labs: { ...prev.labs, ...(partial.labs || {}) },
    }));
    setLoadedMeta(meta);
    setMode("manual");
  };

  const submit = async () => {
    setLoading(true);
    setResult(null);
    const r = await runDiagnosis({ ...form, patientId: form.patientId || "P-001" });
    setResult(r);
    setLoading(false);
  };

  return (
    <section id="demo" className="py-24 px-6 relative">
      <div className="max-w-7xl mx-auto">
        <motion.div initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} className="text-center mb-14">
          <h2 className="text-5xl md:text-6xl font-black tracking-tight mb-4">Run a <span className="gradient-text">Live Diagnosis</span></h2>
          <p className="text-muted-foreground text-lg max-w-2xl mx-auto">Fill in a patient record — our AI doctor will diagnose, prescribe, and audit in real time</p>
        </motion.div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* FORM */}
          <div className="glass rounded-2xl p-6 space-y-3">
            {/* Mode switcher */}
            <div className="flex p-1 rounded-xl bg-muted/40 border border-border/50">
              {([
                { id: "manual", label: "Manual Entry", icon: Pencil },
                { id: "upload", label: "Upload Patient File", icon: Upload },
              ] as const).map((t) => {
                const Icon = t.icon;
                const active = mode === t.id;
                return (
                  <button
                    key={t.id}
                    onClick={() => setMode(t.id)}
                    className={`flex-1 flex items-center justify-center gap-2 py-2.5 rounded-lg text-sm font-medium transition-all ${
                      active ? "gradient-bg text-primary-foreground shadow-lg" : "text-muted-foreground hover:text-foreground"
                    }`}
                  >
                    <Icon size={14} /> {t.label}
                  </button>
                );
              })}
            </div>

            {/* Backend status indicator */}
            {backendOnline !== null && (
              <div className={`flex items-center gap-2 px-3 py-2 rounded-lg text-xs font-medium border ${
                backendOnline
                  ? "bg-success/10 border-success/30 text-success"
                  : "bg-warning/10 border-warning/30 text-warning"
              }`}>
                <div className={`w-2 h-2 rounded-full animate-pulse ${backendOnline ? "bg-success" : "bg-warning"}`} />
                {backendOnline
                  ? "🟢 Python backend connected — running real agents"
                  : "🟡 Backend offline — using mock simulation"}
              </div>
            )}

            {mode === "upload" ? (
              <PatientUpload onLoaded={handleLoaded} />
            ) : (
              <>
                {loadedMeta && (
                  <motion.div
                    initial={{ opacity: 0, y: -6 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="flex items-center justify-between gap-3 px-4 py-3 rounded-xl border border-primary/40 bg-primary/10 text-sm"
                  >
                    <div className="flex items-center gap-2 min-w-0">
                      <FileText size={16} className="text-primary shrink-0" />
                      <span className="truncate">
                        <span className="font-semibold">📂 Loaded from file:</span> {loadedMeta.filename} ·{" "}
                        <span className="font-mono text-primary">{loadedMeta.patientId}</span> · Click any field to edit
                      </span>
                    </div>
                    <button onClick={() => setLoadedMeta(null)} className="text-muted-foreground hover:text-foreground shrink-0">
                      <X size={14} />
                    </button>
                  </motion.div>
                )}
            <Section icon={User} title="Demographics" emoji="🧑" open={open.A} toggle={() => setOpen({ ...open, A: !open.A })}>
              <div className="grid grid-cols-2 gap-3">
                <Field label="Patient ID"><input className={inputCls} placeholder="P-001" value={form.patientId} onChange={(e) => update("patientId", e.target.value)} /></Field>
                <Field label="Age"><input type="number" min={1} max={120} className={inputCls} value={form.age} onChange={(e) => update("age", +e.target.value)} /></Field>
                <Field label="Gender" full>
                  <div className="flex rounded-lg border border-border overflow-hidden">
                    {(["Male","Female","Other"] as const).map(g => (
                      <button key={g} type="button" onClick={() => update("gender", g)}
                        className={`flex-1 py-2 text-sm transition ${form.gender === g ? "gradient-bg text-primary-foreground" : "hover:bg-foreground/5"}`}>{g}</button>
                    ))}
                  </div>
                </Field>
                <Field label="Chief Complaint" full>
                  <textarea rows={3} className={inputCls + " resize-none"} value={form.chiefComplaint} onChange={(e) => update("chiefComplaint", e.target.value)} />
                </Field>
              </div>
            </Section>

            <Section icon={Heart} title="Vitals" emoji="💓" open={open.B} toggle={() => setOpen({ ...open, B: !open.B })}>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                <VitalCard label="BP Sys" unit="mmHg" value={form.vitals.bp_systolic} onChange={v => updateNested("vitals","bp_systolic",v)} range={VITAL_RANGES.bp_systolic} />
                <VitalCard label="BP Dia" unit="mmHg" value={form.vitals.bp_diastolic} onChange={v => updateNested("vitals","bp_diastolic",v)} range={VITAL_RANGES.bp_diastolic} />
                <VitalCard label="Heart Rate" unit="bpm" value={form.vitals.heart_rate} onChange={v => updateNested("vitals","heart_rate",v)} range={VITAL_RANGES.heart_rate} />
                <VitalCard label="Temp" unit="°C" step={0.1} value={form.vitals.temperature} onChange={v => updateNested("vitals","temperature",v)} range={VITAL_RANGES.temperature} />
                <VitalCard label="SpO2" unit="%" value={form.vitals.spo2} onChange={v => updateNested("vitals","spo2",v)} range={VITAL_RANGES.spo2} />
                <VitalCard label="Resp Rate" unit="/min" value={form.vitals.respiratory_rate} onChange={v => updateNested("vitals","respiratory_rate",v)} range={VITAL_RANGES.respiratory_rate} />
              </div>
            </Section>

            <Section icon={FlaskConical} title="Lab Results" emoji="🧪" open={open.C} toggle={() => setOpen({ ...open, C: !open.C })}>
              <div className="space-y-1.5">
                {[
                  ["troponin_i","Troponin I","ng/mL",0.1],
                  ["bnp","BNP","pg/mL",1],
                  ["creatinine","Creatinine","mg/dL",0.1],
                  ["glucose","Glucose","mg/dL",1],
                  ["wbc","WBC","×10³/μL",0.1],
                  ["hemoglobin","Hemoglobin","g/dL",0.1],
                ].map(([k,l,u,s]) => (
                  <div key={k as string} className="flex items-center justify-between gap-3 py-1.5 border-b border-border/40 last:border-0">
                    <span className="text-sm font-medium">{l}</span>
                    <div className="flex items-center gap-2">
                      <input type="number" step={s as number} className={inputCls + " w-24 text-right"} value={(form.labs as any)[k as string]} onChange={(e) => updateNested("labs", k as string, +e.target.value)} />
                      <span className="text-xs text-muted-foreground w-16">{u}</span>
                    </div>
                  </div>
                ))}
              </div>
            </Section>

            <Section icon={ClipboardList} title="Clinical History" emoji="📋" open={open.D} toggle={() => setOpen({ ...open, D: !open.D })}>
              <div className="space-y-3">
                <TagInput label="Known Allergies" tags={form.allergies} onChange={(t) => update("allergies", t)} accent="danger" />
                <TagInput label="Current Medications" tags={form.medications} onChange={(t) => update("medications", t)} accent="primary" />
              </div>
            </Section>

            <Section icon={Settings} title="Agent Settings" emoji="⚙️" open={open.E} toggle={() => setOpen({ ...open, E: !open.E })}>
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <div>
                    <div className="text-sm font-medium">Schema Drift</div>
                    <div className="text-xs text-muted-foreground">Randomly renames clinical keys to test robustness</div>
                  </div>
                  <button onClick={() => update("driftEnabled", !form.driftEnabled)}
                    className={`w-11 h-6 rounded-full p-0.5 transition ${form.driftEnabled ? "gradient-bg" : "bg-muted"}`}>
                    <div className={`w-5 h-5 rounded-full bg-white transition-transform ${form.driftEnabled ? "translate-x-5" : ""}`} />
                  </button>
                </div>
                {form.driftEnabled && (
                  <div>
                    <div className="flex justify-between text-xs mb-1">
                      <span className="text-muted-foreground">Drift Probability</span>
                      <span className="font-mono text-secondary">{form.driftProbability}%</span>
                    </div>
                    <input type="range" min={0} max={100} value={form.driftProbability} onChange={(e) => update("driftProbability", +e.target.value)} className="w-full accent-primary" />
                  </div>
                )}
                <Field label="Seed">
                  <input type="number" className={inputCls + " w-32"} value={form.seed} onChange={(e) => update("seed", +e.target.value)} />
                </Field>
              </div>
            </Section>
              </>
            )}

            <button
              onClick={submit}
              disabled={loading}
              className="w-full mt-4 py-4 rounded-xl font-bold text-primary-foreground gradient-bg hover:scale-[1.01] transition-transform disabled:opacity-70 shadow-[0_0_40px_hsl(239_84%_67%/0.4)] flex items-center justify-center gap-2"
            >
              {loading ? <><Loader2 className="animate-spin" size={20} />Doctor agent is thinking...</> : <><Stethoscope size={20} />🩺 Run Diagnosis</>}
            </button>
          </div>

          {/* RESULTS */}
          <div className="lg:sticky lg:top-6 lg:self-start">
            <ResultsPanel result={result} loading={loading} form={form} />
          </div>
        </div>
      </div>
    </section>
  );
};

const inputCls = "w-full px-3 py-2 rounded-lg bg-background/60 border border-border focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/30 text-sm transition";

const Field = ({ label, children, full }: { label: string; children: React.ReactNode; full?: boolean }) => (
  <div className={full ? "col-span-2" : ""}>
    <label className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-1 block">{label}</label>
    {children}
  </div>
);

const Section = ({ icon: Icon, title, emoji, open, toggle, children }: any) => (
  <div className="border border-border/50 rounded-xl overflow-hidden bg-background/30">
    <button onClick={toggle} className="w-full flex items-center justify-between px-4 py-3 hover:bg-foreground/5 transition">
      <div className="flex items-center gap-2">
        <span className="text-lg">{emoji}</span>
        <Icon size={16} className="text-secondary" />
        <span className="font-semibold text-sm">{title}</span>
      </div>
      <ChevronDown size={16} className={`transition-transform ${open ? "rotate-180" : ""}`} />
    </button>
    <AnimatePresence initial={false}>
      {open && (
        <motion.div initial={{ height: 0 }} animate={{ height: "auto" }} exit={{ height: 0 }} className="overflow-hidden">
          <div className="px-4 pb-4">{children}</div>
        </motion.div>
      )}
    </AnimatePresence>
  </div>
);

const VitalCard = ({ label, unit, value, onChange, range, step = 1 }: { label: string; unit: string; value: number; onChange: (v: number) => void; range: number[]; step?: number }) => {
  const out = value < range[0] || value > range[1];
  return (
    <div className={`rounded-lg p-2.5 border transition-all ${out ? "border-danger/60 bg-danger/5 animate-pulse" : "border-border bg-background/40"}`}>
      <div className="text-[10px] uppercase tracking-wider text-muted-foreground">{label}</div>
      <div className="flex items-baseline gap-1 mt-0.5">
        <input type="number" step={step} value={value} onChange={(e) => onChange(+e.target.value)}
          className="bg-transparent text-lg font-bold w-full outline-none" style={{ color: out ? "hsl(var(--danger))" : undefined }} />
        <span className="text-[10px] text-muted-foreground">{unit}</span>
      </div>
    </div>
  );
};

const TagInput = ({ label, tags, onChange, accent }: { label: string; tags: string[]; onChange: (t: string[]) => void; accent: "danger" | "primary" }) => {
  const [v, setV] = useState("");
  const accentClass = accent === "danger" ? "bg-danger/15 text-danger border-danger/30" : "bg-primary/15 text-primary border-primary/30";
  const onKey = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter" && v.trim()) {
      e.preventDefault();
      if (!tags.includes(v.trim())) onChange([...tags, v.trim()]);
      setV("");
    }
  };
  return (
    <div>
      <label className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-1 block">{label}</label>
      <div className="flex flex-wrap gap-1.5 p-2 rounded-lg border border-border bg-background/40 min-h-[44px]">
        {tags.map(t => (
          <span key={t} className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-xs border ${accentClass}`}>
            {t}
            <button onClick={() => onChange(tags.filter(x => x !== t))}><X size={10} /></button>
          </span>
        ))}
        <input className="bg-transparent flex-1 outline-none text-sm min-w-[80px]" placeholder="Type & press Enter…" value={v} onChange={(e) => setV(e.target.value)} onKeyDown={onKey} />
      </div>
    </div>
  );
};
