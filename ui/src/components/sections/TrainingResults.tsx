import { motion } from "framer-motion";
import { LineChart, Line, XAxis, YAxis, ReferenceLine, Tooltip, Legend, ResponsiveContainer, CartesianGrid, Area, AreaChart } from "recharts";
import { ArrowRight } from "lucide-react";

const REWARDS_V1 = [-0.115,-0.0925,-0.045,-0.05,-0.0075,-0.005,0.0238,0.0463,0.1013,0.1238,0.1713,0.19,0.1825,0.2163,0.2075,0.2125,0.21,0.27,0.2225,0.2725,0.2325,0.225,0.195,0.2075,0.25,0.2263,0.2375,0.2713,0.2275,0.2175];
const REWARDS_V2 = [-0.2175,-0.2213,-0.2238,-0.2313,-0.1613,-0.1775,-0.1275,-0.0863,-0.0913,-0.0775,-0.0238,-0.055,0.0063,-0.0075,-0.0063,0.0188,0.0225,0.045,0.0413,0.0088,0.0338,0.0338,0.02,0.0263,0.07,0.0088,0.0375,0.0125,0.0638,0.0175];
const DATA = REWARDS_V1.map((r, i) => ({ step: (i + 1) * 10, v1: r, v2: REWARDS_V2[i] }));

export const TrainingResults = () => {
  return (
    <section className="py-24 px-6 relative">
      <div className="max-w-7xl mx-auto">
        <motion.div initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} className="text-center mb-14">
          <h2 className="text-5xl md:text-6xl font-black tracking-tight mb-4">GRPO <span className="gradient-text">Training Results</span></h2>
          <p className="text-muted-foreground text-lg">300 steps · Qwen2.5-3B + LoRA · Kaggle T4</p>
        </motion.div>

        <div className="grid grid-cols-1 lg:grid-cols-[1.5fr_1fr] gap-6 mb-8">
          <motion.div initial={{ opacity: 0, y: 30 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }}
            className="glass rounded-2xl p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="font-bold text-lg">Reward Curve</h3>
              <div className="flex items-center gap-2 text-xs text-muted-foreground">
                <span className="h-2 w-2 rounded-full bg-primary" />Mean reward / 10 steps
              </div>
            </div>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={DATA} margin={{ top: 20, right: 20, bottom: 10, left: 0 }}>
                  <defs>
                    <linearGradient id="rewardGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="hsl(239 84% 67%)" stopOpacity={0.6} />
                      <stop offset="100%" stopColor="hsl(239 84% 67%)" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid stroke="hsl(var(--border))" strokeDasharray="3 3" vertical={false} />
                  <XAxis dataKey="step" stroke="hsl(var(--muted-foreground))" fontSize={11} tickLine={false} />
                  <YAxis stroke="hsl(var(--muted-foreground))" fontSize={11} tickLine={false} domain={[-0.2, 0.35]} />
                  <ReferenceLine y={0} stroke="hsl(var(--muted-foreground))" strokeDasharray="4 4" label={{ value: "Baseline", fill: "hsl(var(--muted-foreground))", fontSize: 10, position: "insideTopRight" }} />
                  <Tooltip contentStyle={{ background: "hsl(var(--card))", border: "1px solid hsl(var(--border))", borderRadius: 8, fontSize: 12 }} />
                  <Line type="monotone" dataKey="v1" stroke="hsl(280 84% 67%)" strokeWidth={2} strokeDasharray="5 3" dot={false} name="v1 (hacked)" animationDuration={1500} />
                  <Line type="monotone" dataKey="v2" stroke="hsl(239 84% 67%)" strokeWidth={2.5} dot={false} name="v2 (fixed)" animationDuration={2000} />
                  <Legend />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </motion.div>

          <motion.div initial={{ opacity: 0, y: 30 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true, margin: "-100px" }}
            className="space-y-4">
            <div className="grid grid-cols-1 gap-4">
              <div className="glass rounded-2xl p-5 opacity-60">
                <div className="text-[10px] uppercase tracking-widest text-muted-foreground mb-2">Before Training</div>
                <Stat label="Avg Reward" value="-0.212" tone="muted" />
                <Stat label="Accuracy" value="~35%" tone="muted" />
                <Stat label="Safety" value="0%" tone="muted" />
              </div>
              <div className="flex justify-center"><ArrowRight className="text-success animate-pulse" /></div>
              <div className="rounded-2xl p-5 border border-warning/40 bg-warning/5 mb-2">
                <div className="text-[10px] uppercase tracking-widest text-warning mb-2 font-bold">v1 — Reward Hacked</div>
                <Stat label="Avg Reward" value="+0.20" tone="warning" />
                <Stat label="Accuracy" value="0%" tone="warning" />
                <Stat label="Safety" value="100% (fake)" tone="warning" />
              </div>
              <div className="rounded-2xl p-5 border border-success/40 bg-success/5" style={{ boxShadow: "0 0 40px hsl(var(--success)/0.2)" }}>
                <div className="text-[10px] uppercase tracking-widest text-success mb-2 font-bold">v2 — Fixed</div>
                <Stat label="Avg Reward" value="+0.065" tone="success" />
                <Stat label="Accuracy" value="10%" tone="success" />
                <Stat label="Safety" value="95%" tone="success" />
              </div>
            </div>
          </motion.div>
        </div>

        <motion.div initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }}
          className="flex flex-wrap gap-2 justify-center mb-6">
          {["Qwen2.5-3B-Instruct","300 steps × 2 runs","~219 min on Colab T4","LoRA adapter · 3 reward fixes"].map(c => (
            <span key={c} className="px-3 py-1.5 rounded-full glass text-xs font-mono">{c}</span>
          ))}
        </motion.div>
      </div>
    </section>
  );
};

const Stat = ({ label, value, tone }: { label: string; value: string; tone: "muted" | "success" }) => (
  <div className="flex items-center justify-between py-2 border-b border-border/30 last:border-0">
    <span className="text-xs text-muted-foreground">{label}</span>
    <span className={`font-mono font-bold text-lg ${tone === "success" ? "text-success" : ""}`}>{value}</span>
  </div>
);
