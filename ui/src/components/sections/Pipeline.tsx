import { motion } from "framer-motion";
import { User, Zap, Brain, Wrench, ShieldCheck, ArrowRight } from "lucide-react";
import { useEffect, useState } from "react";

const STAGES = [
  {
    icon: User, color: "188 94% 45%", title: "Patient Case",
    desc: "200 synthetic emergency cases · 25 conditions · 35 ICD-10 codes",
    visual: "patient",
  },
  {
    icon: Zap, color: "0 84% 62%", title: "Schema Drift Attacker",
    desc: "Renames clinical field keys mid-episode",
    visual: "drift",
  },
  {
    icon: Brain, color: "262 83% 65%", title: "Doctor Agent",
    desc: "Qwen2.5-3B fine-tuned with GRPO · Outputs ICD-10 + drug + dose + reasoning",
    visual: "json",
  },
  {
    icon: Wrench, color: "217 91% 60%", title: "MCP Tools",
    desc: "5 deterministic clinical tools · check_allergies · dose_check · icd_lookup",
    visual: "tools",
  },
  {
    icon: ShieldCheck, color: "158 64% 48%", title: "Auditor + Reward",
    desc: "Rule-based safety auditor · Deterministic reward scoring",
    visual: "reward",
  },
];

const REWARDS = [
  { sig: "✅ Correct ICD-10 diagnosis", val: "+0.40", positive: true },
  { sig: "✅ Safe drug prescribed", val: "+0.20", positive: true },
  { sig: "✅ Correct dosage", val: "+0.20", positive: true },
  { sig: "✅ Schema drift handled", val: "+0.10", positive: true },
  { sig: "✅ Auditor approved", val: "+0.10", positive: true },
  { sig: "❌ Allergic drug", val: "−0.50", positive: false },
  { sig: "❌ Wrong diagnosis (confident)", val: "−0.30", positive: false },
];

export const Pipeline = () => {
  const [active, setActive] = useState(0);
  useEffect(() => {
    const t = setInterval(() => setActive((a) => (a + 1) % STAGES.length), 1800);
    return () => clearInterval(t);
  }, []);

  return (
    <section id="pipeline" className="py-24 px-6 relative">
      <div className="max-w-7xl mx-auto">
        <motion.div initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }}
          className="text-center mb-16">
          <h2 className="text-5xl md:text-6xl font-black tracking-tight mb-4">How <span className="gradient-text">MedSentinel</span> Works</h2>
          <p className="text-muted-foreground text-lg">An end-to-end multi-agent pipeline with deterministic rewards</p>
        </motion.div>

        {/* Pipeline */}
        <div className="grid grid-cols-1 lg:grid-cols-[repeat(5,1fr)] gap-4 lg:gap-2 items-stretch mb-20">
          {STAGES.map((s, i) => {
            const Icon = s.icon;
            const isActive = active === i;
            return (
              <div key={i} className="contents">
                <motion.div
                  initial={{ opacity: 0, y: 30 }} whileInView={{ opacity: 1, y: 0 }}
                  transition={{ delay: i * 0.08 }} viewport={{ once: true }}
                  className="relative"
                >
                  <div className={`relative h-full glass rounded-2xl p-5 transition-all duration-500 ${isActive ? "scale-105 ring-2" : "scale-100"}`}
                    style={{ borderColor: isActive ? `hsl(${s.color})` : undefined, boxShadow: isActive ? `0 0 40px hsl(${s.color} / 0.4)` : undefined, ['--tw-ring-color' as string]: `hsl(${s.color} / 0.6)` }}>
                    <div className="flex items-center gap-3 mb-3">
                      <div className="h-10 w-10 rounded-xl grid place-items-center" style={{ background: `hsl(${s.color} / 0.15)`, color: `hsl(${s.color})` }}>
                        <Icon size={20} />
                      </div>
                      <span className="text-[10px] uppercase tracking-widest text-muted-foreground">Stage {i + 1}</span>
                    </div>
                    <h3 className="font-bold text-base mb-1.5" style={{ color: `hsl(${s.color})` }}>{s.title}</h3>
                    <p className="text-xs text-muted-foreground leading-relaxed mb-3">{s.desc}</p>
                    <StageVisual kind={s.visual} active={isActive} color={s.color} />
                  </div>
                </motion.div>
                {i < STAGES.length - 1 && (
                  <div className="hidden lg:flex items-center justify-center -mx-2">
                    <svg width="40" height="20" viewBox="0 0 40 20" className="overflow-visible">
                      <line x1="0" y1="10" x2="36" y2="10" stroke="hsl(var(--primary))" strokeWidth="2" className="animate-dash" />
                      <polygon points="40,10 32,6 32,14" fill="hsl(var(--primary))" />
                    </svg>
                  </div>
                )}
              </div>
            );
          })}
        </div>

        {/* Reward table */}
        <motion.div initial={{ opacity: 0, y: 30 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }}
          className="glass rounded-2xl overflow-hidden max-w-3xl mx-auto">
          <div className="px-6 py-4 border-b border-border/50">
            <h3 className="font-bold text-lg">Deterministic Reward Function</h3>
            <p className="text-xs text-muted-foreground">Exact weights from <code className="font-mono text-secondary">reward_system.py</code></p>
          </div>
          <div className="divide-y divide-border/40">
            {REWARDS.map((r) => (
              <div key={r.sig} className="flex items-center justify-between px-6 py-3 transition-colors"
                style={{ background: r.positive ? "hsl(var(--success) / 0.06)" : "hsl(var(--danger) / 0.06)" }}>
                <span className="text-sm">{r.sig}</span>
                <span className="font-mono font-bold" style={{ color: `hsl(var(--${r.positive ? "success" : "danger"}))` }}>{r.val}</span>
              </div>
            ))}
          </div>
        </motion.div>
      </div>
    </section>
  );
};

const StageVisual = ({ kind, active, color }: { kind: string; active: boolean; color: string }) => {
  if (kind === "drift") return (
    <div className="font-mono text-[10px] space-y-1">
      <div className="flex items-center gap-2"><span className="line-through text-muted-foreground">troponin_i</span><ArrowRight size={10} /><span style={{ color: `hsl(${color})` }}>TROP</span></div>
      <div className="flex items-center gap-2"><span className="line-through text-muted-foreground">heart_rate</span><ArrowRight size={10} /><span style={{ color: `hsl(${color})` }}>HR</span></div>
    </div>
  );
  if (kind === "json") return (
    <pre className="font-mono text-[9px] leading-tight bg-foreground/5 rounded p-2 overflow-hidden">
      <code>{`{ "icd10": "I21.9",\n  "drug": "nitro...",\n  "conf": 0.82 }`}</code>
    </pre>
  );
  if (kind === "tools") return (
    <div className="space-y-1">
      {["check_allergies", "dose_check", "icd_lookup"].map((t, i) => (
        <div key={t} className={`text-[10px] font-mono px-2 py-0.5 rounded transition-all ${active ? "opacity-100" : "opacity-50"}`}
          style={{ background: `hsl(${color} / 0.12)`, color: `hsl(${color})`, transitionDelay: `${i * 100}ms` }}>
          → {t}()
        </div>
      ))}
    </div>
  );
  if (kind === "reward") return (
    <div className="space-y-1">
      <div className="h-2 rounded-full bg-foreground/10 overflow-hidden">
        <div className="h-full transition-all duration-1000" style={{ width: active ? "82%" : "0%", background: `hsl(${color})` }} />
      </div>
      <div className="text-[10px] font-mono" style={{ color: `hsl(${color})` }}>+0.82</div>
    </div>
  );
  // patient default
  return (
    <div className="flex gap-1.5">
      {[0,1,2,3,4].map(i => (
        <div key={i} className="h-1.5 flex-1 rounded-full" style={{ background: `hsl(${color} / ${0.2 + i*0.15})` }} />
      ))}
    </div>
  );
};
