import { Github, ExternalLink } from "lucide-react";

export const Footer = () => (
  <footer className="relative pt-16 pb-8 px-6 border-t border-border/40 mt-16">
    <svg className="absolute top-0 left-0 right-0 h-12 w-full" viewBox="0 0 1200 40" preserveAspectRatio="none">
      <path d="M0 20 L200 20 L220 8 L240 32 L260 4 L280 36 L300 20 L500 20 L520 8 L540 32 L560 4 L580 36 L600 20 L800 20 L820 12 L840 28 L860 6 L880 34 L900 20 L1200 20"
        fill="none" stroke="hsl(var(--primary))" strokeWidth="1.5" className="ecg-line" />
    </svg>
    <div className="max-w-7xl mx-auto grid grid-cols-1 md:grid-cols-3 gap-8 items-center">
      <div>
        <div className="text-2xl font-black gradient-text">MedSentinel</div>
        <p className="text-xs text-muted-foreground mt-1">Built for Hackathon 2025</p>
      </div>
      <div className="text-center text-xs text-muted-foreground">
        MIT License · <span className="text-warning">Research / Education Only</span> · Not for Clinical Use
      </div>
      <div className="flex gap-3 md:justify-end">
        <a href="https://github.com" target="_blank" rel="noreferrer" className="px-4 py-2 rounded-lg glass hover:scale-105 transition flex items-center gap-2 text-xs"><Github size={14} />GitHub</a>
        <a href="https://huggingface.co" target="_blank" rel="noreferrer" className="px-4 py-2 rounded-lg glass hover:scale-105 transition flex items-center gap-2 text-xs">🤗 HuggingFace<ExternalLink size={10} /></a>
      </div>
    </div>
  </footer>
);
