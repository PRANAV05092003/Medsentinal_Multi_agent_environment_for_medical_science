import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { useMemo, useRef, useEffect, useState } from "react";
import * as THREE from "three";
import { useTheme } from "./ThemeProvider";

const PARTICLE_COUNT_DESKTOP = 220;
const PARTICLE_COUNT_MOBILE = 90;

function ParticleField({ count, color, depth, speed }: { count: number; color: string; depth: number; speed: number }) {
  const ref = useRef<THREE.InstancedMesh>(null!);
  const linesRef = useRef<THREE.LineSegments>(null!);
  const mouse = useRef({ x: 0, y: 0 });
  const { viewport } = useThree();

  const positions = useMemo(() => {
    const arr = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      arr[i * 3] = (Math.random() - 0.5) * 18;
      arr[i * 3 + 1] = (Math.random() - 0.5) * 12;
      arr[i * 3 + 2] = (Math.random() - 0.5) * 4 - depth;
    }
    return arr;
  }, [count, depth]);

  const linePositions = useMemo(() => new Float32Array(count * 6 * 3), [count]);

  useEffect(() => {
    const onMove = (e: MouseEvent) => {
      mouse.current.x = (e.clientX / window.innerWidth) * 2 - 1;
      mouse.current.y = -((e.clientY / window.innerHeight) * 2 - 1);
    };
    window.addEventListener("mousemove", onMove);
    return () => window.removeEventListener("mousemove", onMove);
  }, []);

  useFrame((state) => {
    if (!ref.current) return;
    const t = state.clock.getElapsedTime() * speed;
    const dummy = new THREE.Object3D();
    const linePts: number[] = [];
    const mx = mouse.current.x * 0.6;
    const my = mouse.current.y * 0.4;

    for (let i = 0; i < count; i++) {
      const x = positions[i * 3] + Math.sin(t + i) * 0.3 + mx;
      const y = positions[i * 3 + 1] + Math.cos(t * 0.7 + i) * 0.3 + my;
      const z = positions[i * 3 + 2];
      dummy.position.set(x, y, z);
      const s = 0.04 + Math.sin(t * 2 + i) * 0.015;
      dummy.scale.setScalar(s);
      dummy.updateMatrix();
      ref.current.setMatrixAt(i, dummy.matrix);

      // line connections
      for (let j = i + 1; j < Math.min(i + 4, count); j++) {
        const x2 = positions[j * 3] + Math.sin(t + j) * 0.3 + mx;
        const y2 = positions[j * 3 + 1] + Math.cos(t * 0.7 + j) * 0.3 + my;
        const z2 = positions[j * 3 + 2];
        const d = (x - x2) ** 2 + (y - y2) ** 2 + (z - z2) ** 2;
        if (d < 1.6) {
          linePts.push(x, y, z, x2, y2, z2);
        }
      }
    }
    ref.current.instanceMatrix.needsUpdate = true;

    if (linesRef.current) {
      const arr = new Float32Array(linePts);
      linesRef.current.geometry.setAttribute("position", new THREE.BufferAttribute(arr, 3));
      linesRef.current.geometry.attributes.position.needsUpdate = true;
      linesRef.current.geometry.computeBoundingSphere();
    }

    ref.current.rotation.y = t * 0.05;
  });

  return (
    <>
      <instancedMesh ref={ref} args={[undefined, undefined, count]}>
        <sphereGeometry args={[1, 8, 8]} />
        <meshBasicMaterial color={color} transparent opacity={0.85} />
      </instancedMesh>
      <lineSegments ref={linesRef}>
        <bufferGeometry />
        <lineBasicMaterial color={color} transparent opacity={0.18} />
      </lineSegments>
    </>
  );
}

function PulseRings({ color }: { color: string }) {
  const rings = useRef<THREE.Mesh[]>([]);
  useFrame((state) => {
    const t = state.clock.getElapsedTime();
    rings.current.forEach((m, i) => {
      if (!m) return;
      const phase = ((t + i * 1.3) % 4) / 4;
      const s = 0.2 + phase * 6;
      m.scale.set(s, s, 1);
      (m.material as THREE.MeshBasicMaterial).opacity = (1 - phase) * 0.4;
    });
  });
  return (
    <>
      {[0, 1, 2].map((i) => (
        <mesh key={i} ref={(el) => (rings.current[i] = el!)} position={[0, 0, -2]}>
          <ringGeometry args={[0.95, 1, 64]} />
          <meshBasicMaterial color={color} transparent opacity={0.4} side={THREE.DoubleSide} />
        </mesh>
      ))}
    </>
  );
}

export const NeuralBackground = () => {
  const { theme } = useTheme();
  const [count, setCount] = useState(PARTICLE_COUNT_DESKTOP);

  useEffect(() => {
    const update = () => setCount(window.innerWidth < 768 ? PARTICLE_COUNT_MOBILE : PARTICLE_COUNT_DESKTOP);
    update();
    window.addEventListener("resize", update);
    return () => window.removeEventListener("resize", update);
  }, []);

  const primary = theme === "dark" ? "#6366f1" : "#FF5900";
  const secondary = theme === "dark" ? "#06b6d4" : "#FF8237";
  const bgGradient = theme === "dark"
    ? "radial-gradient(ellipse at top, rgba(99,102,241,0.18), transparent 60%), #0a0a0f"
    : "radial-gradient(ellipse at top, rgba(255,89,0,0.12), transparent 60%), #FFFBDC";

  return (
    <div className="fixed inset-0 -z-10 pointer-events-none">
      <div className="absolute inset-0" style={{ background: bgGradient }} />
      <Canvas camera={{ position: [0, 0, 6], fov: 60 }} dpr={[1, 1.5]} frameloop="always">
        <ParticleField count={Math.floor(count * 0.6)} color={primary} depth={0} speed={0.15} />
        <ParticleField count={Math.floor(count * 0.4)} color={secondary} depth={2} speed={0.08} />
        <PulseRings color={primary} />
      </Canvas>
      <div className="absolute inset-0 grid-bg opacity-30" />
    </div>
  );
};
