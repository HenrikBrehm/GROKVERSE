"use client";

import { OrbitControls } from "@react-three/drei";
import { useLayoutEffect, useMemo, useRef } from "react";
import * as THREE from "three";

import { lerpCoords } from "@/lib/data";
import type { LoadedRun } from "@/lib/types";

export function EmbeddingView3D({ run, stepF, autoRotate = true }: {
  run: LoadedRun; stepF: number; autoRotate?: boolean;
}) {
  const p = run.meta.coords_shape[1];
  const geom = useRef<THREE.BufferGeometry>(null);

  const colors = useMemo(() => {
    const arr = new Float32Array(p * 3);
    const col = new THREE.Color();
    for (let i = 0; i < p; i++) {
      col.setHSL(i / p, 0.7, 0.55);
      arr[i * 3] = col.r;
      arr[i * 3 + 1] = col.g;
      arr[i * 3 + 2] = col.b;
    }
    return arr;
  }, [p]);

  const positions = useMemo(() => new Float32Array(p * 3), [p]);

  // Interpolate + upload outside render: mutating the memoized buffer during
  // render would be impure and can tear under concurrent rendering.
  useLayoutEffect(() => {
    const g = geom.current;
    if (!g) return;
    positions.set(lerpCoords(run, stepF));
    const pos = g.getAttribute("position") as THREE.BufferAttribute;
    (pos.array as Float32Array).set(positions);
    pos.needsUpdate = true;
  }, [run, stepF, positions]);

  return (
    <>
      <ambientLight intensity={0.7} />
      <points>
        <bufferGeometry ref={geom}>
          <bufferAttribute attach="attributes-position" args={[positions, 3]} />
          <bufferAttribute attach="attributes-color" args={[colors, 3]} />
        </bufferGeometry>
        <pointsMaterial vertexColors size={0.07} sizeAttenuation transparent opacity={0.95} />
      </points>
      <OrbitControls enablePan={false} autoRotate={autoRotate} autoRotateSpeed={0.4} />
    </>
  );
}
