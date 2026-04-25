import { NeuralBackground } from "@/components/NeuralBackground";
import { ThemeToggle } from "@/components/ThemeToggle";
import { Hero } from "@/components/sections/Hero";
import { Pipeline } from "@/components/sections/Pipeline";
import { InteractiveDemo } from "@/components/sections/InteractiveDemo";
import { TrainingResults } from "@/components/sections/TrainingResults";
import { Architecture } from "@/components/sections/Architecture";
import { Footer } from "@/components/sections/Footer";

const Index = () => {
  return (
    <div className="relative min-h-screen">
      <NeuralBackground />
      <ThemeToggle />
      <main>
        <Hero />
        <Pipeline />
        <InteractiveDemo />
        <TrainingResults />
        <Architecture />
      </main>
      <Footer />
    </div>
  );
};

export default Index;
