import { Moon, Sun } from "lucide-react";
import { useTheme } from "./ThemeProvider";
import { motion, AnimatePresence } from "framer-motion";

export const ThemeToggle = () => {
  const { theme, toggle } = useTheme();
  return (
    <button
      onClick={toggle}
      aria-label="Toggle theme"
      className="fixed top-5 right-5 z-50 h-11 w-11 grid place-items-center rounded-full text-white hover:scale-110 transition-transform shadow-lg"
      style={{ background: theme === "dark" ? "#6366f1" : "#FF5900", boxShadow: "var(--gradient-glow)" }}
    >
      <AnimatePresence mode="wait" initial={false}>
        <motion.span
          key={theme}
          initial={{ rotate: -90, opacity: 0 }}
          animate={{ rotate: 0, opacity: 1 }}
          exit={{ rotate: 90, opacity: 0 }}
          transition={{ duration: 0.25 }}
          className="text-foreground"
        >
          {theme === "dark" ? <Moon size={18} /> : <Moon size={18} />}
        </motion.span>
      </AnimatePresence>
    </button>
  );
};
