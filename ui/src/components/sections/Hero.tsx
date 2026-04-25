import { motion } from "framer-motion";
import { ArrowRight, Stethoscope, Activity } from "lucide-react";
import { CountUp } from "../CountUp";

export const Hero = () => {
  const scrollTo = (id: string) => document.getElementById(id)?.scrollIntoView({ behavior: "smooth" });
  const stats = [
    { value: 200, suffix: "", label: "Patient Cases" },
    { value: 25, suffix: "+", label: "Conditions Covered" },
    { value: 1.0, suffix: "", prefix: "+", label: "Max Reward", decimals: 1 },
    { value: 0, suffix: "", label: "LLM Judges" },
  ];

  return (
    <section className="relative min-h-screen flex items-center justify-center px-6 py-24">
      <div className="max-w-5xl mx-auto text-center relative z-10">
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.6 }}
          className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full glass text-xs font-medium mb-8">
          <Activity size={14} className="text-secondary" />
          <span className="text-foreground/80">Multi-Agent Medical RL · Hackathon 2025</span>
        </motion.div>

        <motion.h1
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.1 }}
          className="text-7xl md:text-9xl font-black tracking-tighter gradient-text mb-6 leading-none"
        >
          MedSentinel
        </motion.h1>

        <motion.p
          initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.6, delay: 0.25 }}
          className="text-2xl md:text-3xl font-light text-foreground/90 mb-5 text-balance"
        >
          AI that learns to save lives — even when the data fights back
        </motion.p>

        <motion.p
          initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.6, delay: 0.35 }}
          className="text-base md:text-lg text-muted-foreground max-w-2xl mx-auto mb-10 text-balance"
        >
          A doctor AI agent trained with GRPO learns to diagnose patients under adversarial schema drift attacks.
          Fully deterministic rewards. No LLM judge.
        </motion.p>

        <motion.div
          initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.6, delay: 0.45 }}
          className="flex flex-col sm:flex-row gap-4 justify-center mb-16"
        >
          <button onClick={() => scrollTo("demo")}
            className="group relative px-8 py-3.5 rounded-xl font-semibold text-primary-foreground gradient-bg hover:scale-105 transition-transform shadow-[0_0_40px_hsl(239_84%_67%/0.4)]">
            <span className="flex items-center gap-2"><Stethoscope size={18} />Run a Diagnosis<ArrowRight size={16} className="group-hover:translate-x-1 transition" /></span>
          </button>
          <button onClick={() => scrollTo("pipeline")}
            className="px-8 py-3.5 rounded-xl font-semibold glass hover:bg-foreground/5 transition">
            View Architecture
          </button>
        </motion.div>

        <motion.div
          initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.8, delay: 0.6 }}
          className="grid grid-cols-2 md:grid-cols-4 gap-4 max-w-3xl mx-auto"
        >
          {stats.map((s, i) => (
            <motion.div key={s.label}
              initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5, delay: 0.7 + i * 0.1 }}
              className="glass rounded-2xl p-5">
              <div className="text-3xl md:text-4xl font-bold gradient-text">
                {s.prefix}<CountUp end={s.value} decimals={s.decimals ?? 0} />{s.suffix}
              </div>
              <div className="text-xs text-muted-foreground mt-1 uppercase tracking-wider">{s.label}</div>
            </motion.div>
          ))}
        </motion.div>
      </div>
    </section>
  );
};
