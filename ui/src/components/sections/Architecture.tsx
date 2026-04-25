import { motion } from "framer-motion";
import { useState } from "react";
import { Folder, FileCode, Server, Brain, ShieldCheck, Wrench, Zap, BarChart3, RefreshCw } from "lucide-react";

const FILES = [
  { name: "openenv_server/", icon: Folder, desc: "FastAPI bridge" },
  { name: "  app.py", icon: FileCode, desc: "OpenEnv API endpoints" },
  { name: "environment/", icon: Folder, desc: "MedSentinel env" },
  { name: "  med_env.py", icon: FileCode, desc: "Episode loop" },
  { name: "  schema_drift.py", icon: FileCode, desc: "Adversarial attacker" },
  { name: "agents/", icon: Folder, desc: "" },
  { name: "  doctor_agent.py", icon: FileCode, desc: "Qwen2.5-3B + GRPO" },
  { name: "  auditor.py", icon: FileCode, desc: "Rule-based checks" },
  { name: "tools/", icon: Folder, desc: "MCP clinical tools" },
  { name: "  check_allergies.py", icon: FileCode, desc: "" },
  { name: "  dose_check.py", icon: FileCode, desc: "" },
  { name: "  icd_lookup.py", icon: FileCode, desc: "" },
  { name: "rewards/", icon: Folder, desc: "" },
  { name: "  reward_system.py", icon: FileCode, desc: "Deterministic scoring" },
  { name: "training/", icon: Folder, desc: "" },
  { name: "  grpo_train.py", icon: FileCode, desc: "300-step loop" },
];

const NODES = [
  { id: "env", label: "OpenEnv Server", icon: Server, x: 50, y: 15, color: "188 94% 50%" },
  { id: "med", label: "Med Environment", icon: Zap, x: 50, y: 38, color: "0 84% 60%" },
  { id: "doc", label: "Doctor Agent", icon: Brain, x: 18, y: 60, color: "262 83% 65%" },
  { id: "tool", label: "MCP Tools (×5)", icon: Wrench, x: 50, y: 60, color: "217 91% 60%" },
  { id: "aud", label: "Auditor", icon: ShieldCheck, x: 82, y: 60, color: "158 64% 50%" },
  { id: "rew", label: "Reward System", icon: BarChart3, x: 35, y: 85, color: "38 92% 55%" },
  { id: "loop", label: "GRPO Loop", icon: RefreshCw, x: 65, y: 85, color: "239 84% 67%" },
];

const EDGES: [string, string][] = [
  ["env","med"],["med","doc"],["med","tool"],["med","aud"],
  ["doc","tool"],["doc","aud"],["aud","rew"],["tool","rew"],["rew","loop"],["loop","doc"],
];

export const Architecture = () => {
  const [hover, setHover] = useState<number | null>(null);
  return (
    <section className="py-24 px-6 relative">
      <div className="max-w-7xl mx-auto">
        <motion.div initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} className="text-center mb-14">
          <h2 className="text-5xl md:text-6xl font-black tracking-tight mb-4">System <span className="gradient-text">Architecture</span></h2>
          <p className="text-muted-foreground text-lg">A modular multi-agent system built on OpenEnv + MCP</p>
        </motion.div>

        <div className="grid grid-cols-1 lg:grid-cols-[1fr_1.5fr] gap-6">
          {/* file tree */}
          <motion.div initial={{ opacity: 0, x: -20 }} whileInView={{ opacity: 1, x: 0 }} viewport={{ once: true }}
            className="glass rounded-2xl p-5 font-mono text-xs">
            <div className="flex items-center gap-2 pb-3 mb-3 border-b border-border/40">
              <div className="flex gap-1.5">
                <div className="h-2.5 w-2.5 rounded-full bg-danger" />
                <div className="h-2.5 w-2.5 rounded-full bg-warning" />
                <div className="h-2.5 w-2.5 rounded-full bg-success" />
              </div>
              <span className="text-muted-foreground ml-2">medsentinel/</span>
            </div>
            <div className="space-y-0.5">
              {FILES.map((f, i) => {
                const Icon = f.icon;
                return (
                  <div key={i}
                    onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)}
                    className={`flex items-center gap-2 px-2 py-1 rounded transition-all cursor-pointer ${hover === i ? "bg-primary/15 text-primary" : "hover:bg-foreground/5"}`}>
                    <Icon size={12} className={hover === i ? "text-primary" : "text-muted-foreground"} />
                    <span className="whitespace-pre">{f.name}</span>
                    {f.desc && <span className="text-muted-foreground text-[10px] ml-auto">{f.desc}</span>}
                  </div>
                );
              })}
            </div>
          </motion.div>

          {/* graph */}
          <motion.div initial={{ opacity: 0, x: 20 }} whileInView={{ opacity: 1, x: 0 }} viewport={{ once: true }}
            className="glass rounded-2xl p-5 relative aspect-square lg:aspect-auto min-h-[500px]">
            <svg className="absolute inset-0 w-full h-full" viewBox="0 0 100 100" preserveAspectRatio="none">
              {EDGES.map(([from, to], i) => {
                const a = NODES.find(n => n.id === from)!;
                const b = NODES.find(n => n.id === to)!;
                return (
                  <line key={i} x1={a.x} y1={a.y} x2={b.x} y2={b.y}
                    stroke="hsl(var(--primary))" strokeWidth="0.25" strokeOpacity="0.4"
                    strokeDasharray="1 0.6" className="animate-dash" />
                );
              })}
            </svg>
            {NODES.map(n => {
              const Icon = n.icon;
              return (
                <motion.div key={n.id}
                  initial={{ opacity: 0, scale: 0 }} whileInView={{ opacity: 1, scale: 1 }} viewport={{ once: true }}
                  transition={{ delay: 0.2 + Math.random() * 0.4 }}
                  className="absolute -translate-x-1/2 -translate-y-1/2 group"
                  style={{ left: `${n.x}%`, top: `${n.y}%` }}>
                  <div className="px-3 py-2 rounded-xl glass flex items-center gap-2 hover:scale-110 transition cursor-pointer"
                    style={{ borderColor: `hsl(${n.color} / 0.4)`, boxShadow: `0 0 20px hsl(${n.color} / 0.3)` }}>
                    <div className="h-7 w-7 rounded-lg grid place-items-center" style={{ background: `hsl(${n.color} / 0.15)`, color: `hsl(${n.color})` }}>
                      <Icon size={14} />
                    </div>
                    <span className="text-xs font-semibold whitespace-nowrap">{n.label}</span>
                  </div>
                </motion.div>
              );
            })}
          </motion.div>
        </div>
      </div>
    </section>
  );
};
