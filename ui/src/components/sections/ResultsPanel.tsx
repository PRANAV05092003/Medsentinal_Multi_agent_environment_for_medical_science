import { motion, AnimatePresence } from "framer-motion";
import { useEffect, useState } from "react";
import { AlertTriangle, ShieldCheck, ShieldAlert, ChevronDown, Activity, Pill, Trophy } from "lucide-react";
import { DiagnosisResult, PatientForm } from "@/lib/diagnosisEngine";

const CVLCard = ({ result }: { result: DiagnosisResult }) => {
  const cvl = result.cvl!;
  return (
    <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.5 }}
      className={`glass rounded-2xl p-5 border ${cvl.verified && !cvl.fallback ? "border-purple-500/40" : "border-muted/40"}`}>
      <div className="flex items-center gap-2 mb-3">
        <span className="text-lg">🔐</span>
        <h3 className="font-bold">Clinical Verification Layer (2FA)</h3>
        {cvl.fallback
          ? <span className="ml-auto text-[10px] font-bold px-2 py-1 rounded bg-muted/20 text-muted-foreground">PASS-THROUGH</span>
          : cvl.verified
          ? <span className="ml-auto text-[10px] font-bold px-2 py-1 rounded bg-purple-500/15 text-purple-400">VERIFIED</span>
          : <span className="ml-auto text-[10px] font-bold px-2 py-1 rounded bg-warning/15 text-warning">PENDING</span>
        }
      </div>
      {cvl.fallback ? (
        <div className="text-xs text-muted-foreground">{cvl.notes}</div>
      ) : (
        <>
          {cvl.changes.length > 0 ? (
            <div className="mb-3">
              <div className="text-xs font-semibold text-warning mb-1.5">🔄 {cvl.changes.length} correction(s) made:</div>
              {cvl.changes.map((c, i) => <div key={i} className="text-xs text-foreground/80 pl-3 border-l-2 border-warning/50 mb-1">{c}</div>)}
            </div>
          ) : (
            <div className="flex items-center gap-2 text-xs text-green-400 mb-3">
              <ShieldCheck size={14} />Doctor output verified — no corrections needed
            </div>
          )}
          {cvl.riskFlags.length > 0 && (
            <div className="flex flex-wrap gap-1.5 mb-3">
              {cvl.riskFlags.map(f => <span key={f} className="px-2 py-0.5 rounded bg-danger/15 text-danger text-[10px] font-mono">{f}</span>)}
            </div>
          )}
          {cvl.notes && <div className="text-xs text-muted-foreground">{cvl.notes}</div>}
        </>
      )}
    </motion.div>
  );
};


export const ResultsPanel = ({ result, loading, form }: { result: DiagnosisResult | null; loading: boolean; form: PatientForm }) => {
  if (loading) return <SkeletonResults />;
  if (!result) return <EmptyState />;

  return (
    <motion.div initial={{ opacity: 0, x: 30 }} animate={{ opacity: 1, x: 0 }} className="space-y-4">
      <DriftCard result={result} />
      <DoctorCard result={result} />
      <ToolCallsCard result={result} />
      <AuditorCard result={result} />
      <RewardCard result={result} />
      {result.cvl && <CVLCard result={result} />}
      {result._source === "mock" && (
        <div className="flex items-center gap-2 px-4 py-3 rounded-xl border border-warning/40 bg-warning/5 text-warning text-xs">
          ⚠️ Running in mock mode — start the Python backend for real results
        </div>
      )}
    </motion.div>
  );
};

const EmptyState = () => (
  <div className="glass rounded-2xl p-12 text-center min-h-[600px] flex flex-col items-center justify-center">
    <div className="h-16 w-16 rounded-full gradient-bg grid place-items-center mb-4 animate-float">
      <Activity className="text-primary-foreground" />
    </div>
    <h3 className="font-bold text-xl mb-2">Awaiting Patient Data</h3>
    <p className="text-sm text-muted-foreground max-w-xs">Fill in the form and click <span className="text-primary font-semibold">Run Diagnosis</span> to see the multi-agent pipeline in action.</p>
  </div>
);

const SkeletonResults = () => (
  <div className="space-y-4">
    {[1,2,3,4,5].map(i => (
      <div key={i} className="glass rounded-2xl p-5 h-32 relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-r from-transparent via-foreground/5 to-transparent animate-shimmer" style={{ backgroundSize: "200% 100%" }} />
        <div className="h-3 w-1/3 bg-foreground/10 rounded mb-3" />
        <div className="h-3 w-2/3 bg-foreground/10 rounded mb-2" />
        <div className="h-3 w-1/2 bg-foreground/10 rounded" />
      </div>
    ))}
  </div>
);

const DriftCard = ({ result }: { result: DiagnosisResult }) => (
  <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}
    className={`glass rounded-2xl p-5 ${result.drift.occurred ? "border-warning/40" : "border-success/40"}`}>
    <div className="flex items-center gap-2 mb-3">
      <span className="text-lg">🔀</span>
      <h3 className="font-bold">Schema Drift Report</h3>
    </div>
    {result.drift.occurred ? (
      <>
        <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-warning/10 text-warning text-sm mb-3">
          <AlertTriangle size={14} />Drift detected — {result.drift.renames.length} key(s) renamed mid-episode
        </div>
        <div className="space-y-1.5">
          {result.drift.renames.map((r, i) => (
            <div key={i} className="flex items-center gap-2 font-mono text-xs px-3 py-2 rounded-lg bg-background/50">
              <span className="text-[10px] uppercase text-muted-foreground w-12">{r.section}</span>
              <span className="line-through text-muted-foreground">{r.original}</span>
              <span>→</span>
              <span className="text-warning font-bold">{r.renamed}</span>
            </div>
          ))}
        </div>
      </>
    ) : (
      <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-success/10 text-success text-sm">
        <ShieldCheck size={14} />No schema drift this episode
      </div>
    )}
  </motion.div>
);

const DoctorCard = ({ result }: { result: DiagnosisResult }) => {
  const [expanded, setExpanded] = useState(false);
  const [typed, setTyped] = useState("");
  useEffect(() => {
    if (!expanded) return;
    let i = 0;
    const id = setInterval(() => {
      i += 3;
      setTyped(result.doctor.reasoning.slice(0, i));
      if (i >= result.doctor.reasoning.length) clearInterval(id);
    }, 12);
    return () => clearInterval(id);
  }, [expanded, result.doctor.reasoning]);

  const conf = Math.round(result.doctor.confidence * 100);
  const r = 36, c = 2 * Math.PI * r;

  return (
    <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}
      className="glass rounded-2xl p-5 relative overflow-hidden">
      <div className="absolute inset-0 opacity-30 pointer-events-none" style={{ background: "radial-gradient(circle at 100% 0%, hsl(var(--primary)/0.3), transparent 50%)" }} />
      <div className="relative">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <span className="text-lg">🩺</span>
            <h3 className="font-bold">Doctor Agent Output</h3>
          </div>
          {result.doctor.schemaDriftHandled && (
            <span className="text-[10px] font-bold px-2 py-1 rounded bg-success/15 text-success">DRIFT HANDLED</span>
          )}
        </div>

        <div className="grid grid-cols-[1fr_auto] gap-4 items-center">
          <div>
            <div className="text-3xl font-black gradient-text font-mono">{result.doctor.icd10}</div>
            <div className="text-sm font-medium mt-1">{result.doctor.diagnosisName}</div>
            <div className="flex flex-wrap items-center gap-2 mt-3">
              <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full bg-secondary/15 text-secondary text-xs font-mono">
                <Pill size={12} />{result.doctor.drug.replace("_", " ")}
              </span>
              <span className="px-2.5 py-1 rounded-full bg-primary/15 text-primary text-xs font-mono">{result.doctor.dose}</span>
            </div>
          </div>
          <div className="relative w-20 h-20">
            <svg viewBox="0 0 80 80" className="-rotate-90">
              <circle cx="40" cy="40" r={r} stroke="hsl(var(--muted))" strokeWidth="6" fill="none" />
              <motion.circle cx="40" cy="40" r={r} stroke="url(#confGrad)" strokeWidth="6" fill="none"
                strokeDasharray={c} strokeLinecap="round"
                initial={{ strokeDashoffset: c }}
                animate={{ strokeDashoffset: c - (c * conf) / 100 }}
                transition={{ duration: 1.2, ease: "easeOut" }} />
              <defs>
                <linearGradient id="confGrad" x1="0" x2="1">
                  <stop offset="0%" stopColor="hsl(var(--primary))" />
                  <stop offset="100%" stopColor="hsl(var(--secondary))" />
                </linearGradient>
              </defs>
            </svg>
            <div className="absolute inset-0 grid place-items-center text-sm font-bold">{conf}%</div>
          </div>
        </div>

        <button onClick={() => setExpanded(!expanded)} className="mt-4 flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground">
          <ChevronDown size={12} className={`transition ${expanded ? "rotate-180" : ""}`} />
          {expanded ? "Hide" : "Show"} reasoning
        </button>
        <AnimatePresence>
          {expanded && (
            <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: "auto", opacity: 1 }} exit={{ height: 0, opacity: 0 }}
              className="overflow-hidden">
              <div className="mt-3 p-3 rounded-lg bg-background/60 font-mono text-xs leading-relaxed border border-border">
                {typed}<span className="inline-block w-1.5 h-3 bg-primary animate-pulse ml-0.5" />
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </motion.div>
  );
};

const verdictStyles = {
  safe: { bg: "bg-success/15", text: "text-success", emoji: "🟢" },
  unsafe: { bg: "bg-danger/15", text: "text-danger", emoji: "🔴" },
  warning: { bg: "bg-warning/15", text: "text-warning", emoji: "🟡" },
  drift: { bg: "bg-primary/15", text: "text-primary", emoji: "⚠️" },
};

const ToolCallsCard = ({ result }: { result: DiagnosisResult }) => {
  const [expanded, setExpanded] = useState<number | null>(null);
  return (
    <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}
      className="glass rounded-2xl p-5">
      <div className="flex items-center gap-2 mb-4">
        <span className="text-lg">🔧</span>
        <h3 className="font-bold">MCP Tool Call Log</h3>
        <span className="text-xs text-muted-foreground ml-auto font-mono">{result.toolCalls.length} calls</span>
      </div>
      <div className="space-y-2">
        {result.toolCalls.map((tc, i) => {
          const v = verdictStyles[tc.verdict];
          return (
            <motion.div key={i} initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: i * 0.15 }}
              className="rounded-lg border border-border bg-background/40 overflow-hidden">
              <button onClick={() => setExpanded(expanded === i ? null : i)} className="w-full flex items-center gap-2 px-3 py-2 hover:bg-foreground/5">
                <code className="font-mono text-xs px-2 py-0.5 rounded bg-foreground/10 font-bold">{tc.name}</code>
                <span className="text-[10px] text-muted-foreground truncate flex-1 text-left">{Object.keys(tc.input).slice(0,2).join(", ")}</span>
                <span className={`text-[10px] font-bold px-2 py-0.5 rounded ${v.bg} ${v.text}`}>{v.emoji} {tc.verdict.toUpperCase()}</span>
                <ChevronDown size={12} className={`transition ${expanded === i ? "rotate-180" : ""}`} />
              </button>
              <AnimatePresence>
                {expanded === i && (
                  <motion.div initial={{ height: 0 }} animate={{ height: "auto" }} exit={{ height: 0 }} className="overflow-hidden">
                    <pre className="text-[10px] font-mono p-3 bg-background/60 border-t border-border overflow-x-auto"><code>{JSON.stringify({ input: tc.input, output: tc.output }, null, 2)}</code></pre>
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>
          );
        })}
      </div>
    </motion.div>
  );
};

const AuditorCard = ({ result }: { result: DiagnosisResult }) => (
  <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.3 }}
    className={`rounded-2xl p-5 relative overflow-hidden ${result.auditor.safe ? "bg-success/10 border border-success/40" : "bg-danger/10 border border-danger/40"}`}
    style={{ boxShadow: `0 0 60px hsl(var(--${result.auditor.safe ? "success" : "danger"}) / 0.25)` }}>
    <div className="flex items-center gap-3">
      <motion.div initial={{ scale: 0 }} animate={{ scale: 1 }} transition={{ type: "spring", delay: 0.4 }}
        className={`h-12 w-12 rounded-xl grid place-items-center ${result.auditor.safe ? "bg-success/20 text-success" : "bg-danger/20 text-danger"}`}>
        {result.auditor.safe ? <ShieldCheck size={24} /> : <ShieldAlert size={24} />}
      </motion.div>
      <div>
        <div className="text-[10px] uppercase tracking-widest text-muted-foreground">Auditor Verdict</div>
        <div className={`text-xl font-black ${result.auditor.safe ? "text-success" : "text-danger"}`}>
          {result.auditor.safe ? "✅ SAFE — No violations" : "❌ UNSAFE — Violations detected"}
        </div>
      </div>
    </div>
    {result.auditor.flags.length > 0 && (
      <div className="flex flex-wrap gap-2 mt-3">
        {result.auditor.flags.map(f => (
          <span key={f} className="px-2 py-1 rounded-md bg-danger/20 text-danger text-xs font-mono font-bold">{f}</span>
        ))}
      </div>
    )}
  </motion.div>
);

const RewardCard = ({ result }: { result: DiagnosisResult }) => {
  const total = result.reward.total;
  const tier = total >= 0.6 ? { c: "success", l: "Excellent" } : total >= 0.2 ? { c: "warning", l: "Acceptable" } : { c: "danger", l: "Poor" };
  const pct = Math.max(0, Math.min(100, ((total + 1) / 2) * 100));

  return (
    <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.4 }}
      className="glass rounded-2xl p-5">
      <div className="flex items-center gap-2 mb-4">
        <Trophy size={18} className="text-warning" />
        <h3 className="font-bold">Reward Breakdown</h3>
      </div>

      <div className="flex items-end justify-between mb-3">
        <div>
          <div className="text-5xl font-black font-mono" style={{ color: `hsl(var(--${tier.c}))` }}>
            {total >= 0 ? "+" : ""}{total.toFixed(3)}
          </div>
          <div className="text-xs uppercase tracking-widest font-bold" style={{ color: `hsl(var(--${tier.c}))` }}>{tier.l}</div>
        </div>
        <div className="text-xs text-muted-foreground font-mono">range [-1, +1]</div>
      </div>

      <div className="h-3 rounded-full bg-background/60 overflow-hidden border border-border mb-4 relative">
        <div className="absolute top-0 bottom-0 w-px bg-foreground/30" style={{ left: "50%" }} />
        <motion.div initial={{ width: 0 }} animate={{ width: `${pct}%` }} transition={{ duration: 1.2, ease: "easeOut" }}
          className="h-full" style={{ background: `linear-gradient(90deg, hsl(var(--danger)), hsl(var(--warning)), hsl(var(--success)))` }} />
      </div>

      <div className="space-y-1.5">
        {result.reward.components.map((c, i) => {
          const positive = c.value >= 0;
          return (
            <motion.div key={i} initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.5 + i * 0.08 }}
              className="flex items-center gap-3 text-xs">
              <span className="flex-1 truncate">{c.label}</span>
              <div className="w-24 h-1.5 rounded-full bg-foreground/10 overflow-hidden">
                <motion.div initial={{ width: 0 }} animate={{ width: `${Math.abs(c.value) * 200}%` }} transition={{ delay: 0.6 + i * 0.08, duration: 0.6 }}
                  className="h-full" style={{ background: `hsl(var(--${positive ? "success" : "danger"}))` }} />
              </div>
              <span className={`font-mono font-bold w-12 text-right ${positive ? "text-success" : "text-danger"}`}>
                {positive ? "+" : ""}{c.value.toFixed(2)}
              </span>
            </motion.div>
          );
        })}
      </div>
    </motion.div>
  );
};
