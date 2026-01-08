"use client";

import { useState, useEffect, Suspense } from "react";
import { Brain, Loader2 } from "lucide-react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls, Grid } from "@react-three/drei";
import * as THREE from "three";
import GameContextPanel from "./GameContextPanel";
import CommanderBoard from "./CommanderBoard";
import PitchInputForm from "./PitchInputForm";
import RiskCloud from "./visualizer/RiskCloud";
import { usePitchStore } from "@/store/pitchStore";

// Strike Zone visual component (MLB standard: 17" wide × ~21" tall)
function StrikeZone() {
  const geometry = new THREE.BoxGeometry(1.42, 1.8, 0.02); // feet: 17"/12 = 1.42ft
  const edges = new THREE.EdgesGeometry(geometry);

  return (
    <group position={[0, 2.5, 0]}>
      <lineSegments geometry={edges}>
        <lineBasicMaterial color="#22d3ee" linewidth={2} />
      </lineSegments>
    </group>
  );
}

export default function CommanderDashboard() {
  const { recommendations, bestPitchId, comparePitchId, getComparePitch } = usePitchStore();
  const [selectedPitchId, setSelectedPitchId] = useState<string | null>(null);

  // Auto-select best pitch when recommendations load
  useEffect(() => {
    if (bestPitchId) {
      setSelectedPitchId(bestPitchId);
    }
  }, [bestPitchId]);

  // Find selected pitch data
  const selectedPitch = recommendations.find(
    (p) => p.pitch_id === selectedPitchId
  );

  // Transform scatter data for RiskCloud
  const visualizationPoints =
    selectedPitch?.scatter_sample?.map((pt) => ({
      x: pt.x,
      y: pt.y || 0,
      z: pt.z,
      xRV: pt.xrv,
    })) || [];

  // Get comparison pitch (ghost)
  const ghostPitch = getComparePitch?.();
  const ghostPoints = ghostPitch?.scatter_sample?.map((pt) => ({
    x: pt.x,
    y: pt.y || 0,
    z: pt.z,
    xRV: pt.xrv,
  })) || [];

  // Extract suggested target coordinates (AI-assigned or user-specified)
  const suggestedTarget = selectedPitch?.suggested_target
    ? {
        x: selectedPitch.suggested_target[0],
        z: selectedPitch.suggested_target[1],
      }
    : { x: 0, z: 2.5 }; // Default center

  return (
    <div className="h-screen w-full flex overflow-hidden bg-slate-950">
      {/* Header */}
      <div className="absolute top-0 left-0 right-0 z-40 bg-gradient-to-r from-slate-900 via-blue-900/20 to-purple-900/20 border-b border-slate-800 backdrop-blur">
        <div className="px-6 py-4">
          <div className="flex items-center gap-3">
            <Brain className="w-6 h-6 text-blue-400" />
            <div>
              <h1 className="text-xl font-bold text-white">
                Aegis Commander
              </h1>
              <p className="text-xs text-slate-400">
                Intelligence-Driven Pitch Recommendation System
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Main Layout */}
      <div className="flex w-full pt-20">
        {/* Left Sidebar - Configuration */}
        <div className="w-[380px] flex-shrink-0 h-full overflow-y-auto border-r border-slate-800 bg-slate-900/90 backdrop-blur">
          <div className="p-4 space-y-6">
            {/* Game Context */}
            <GameContextPanel />

            {/* Pitch Arsenal Configuration - Commander Mode */}
            <PitchInputForm mode="commander" />
          </div>
        </div>

        {/* Main Content - Split View */}
        <div className="flex-1 grid grid-cols-12 gap-0 h-full overflow-hidden">
          {/* Left: Commander Board (7/12 columns) */}
          <div className="col-span-7 h-full overflow-y-auto bg-slate-950 p-6 border-r border-slate-800">
            <div className="max-w-4xl mx-auto">
              <div className="mb-4">
                <h2 className="text-lg font-bold text-white mb-1">
                  Strategy Deck
                </h2>
                <p className="text-sm text-slate-400">
                  Click any card to visualize its targeting distribution
                </p>
              </div>

              <CommanderBoard
                onCardClick={(id) => setSelectedPitchId(id)}
                selectedId={selectedPitchId}
              />
            </div>
          </div>

          {/* Right: 3D Visualization (5/12 columns) */}
          <div className="col-span-5 h-full bg-slate-900 relative">
            {/* Header */}
            <div className="absolute top-4 left-4 right-4 z-20 bg-black/60 backdrop-blur-sm px-4 py-3 rounded-lg border border-slate-700">
              <div className="flex items-center justify-between">
                <div className="flex-1">
                  <div className="text-xs text-slate-400 uppercase tracking-wide mb-1">
                    {ghostPitch ? "Tunneling Comparison" : "Simulated Trajectory"}
                  </div>
                  <div className="flex items-center gap-2 mb-1">
                    <div className="text-lg font-bold text-cyan-400">
                      {selectedPitch?.pitch_type || "Select a pitch"}
                    </div>
                    {ghostPitch && (
                      <>
                        <span className="text-slate-500 text-sm">vs</span>
                        <div className="text-lg font-bold text-purple-400">
                          {ghostPitch.pitch_type}
                        </div>
                      </>
                    )}
                  </div>
                  {selectedPitch && suggestedTarget && (
                    <div className="flex items-center gap-2 text-xs">
                      <span className="text-slate-400">Target:</span>
                      <span className="text-yellow-400 font-mono">
                        ({suggestedTarget.x.toFixed(2)}, {suggestedTarget.z.toFixed(2)})
                      </span>
                      <span className="text-slate-500">ft</span>
                    </div>
                  )}
                </div>
                {visualizationPoints.length > 0 && (
                  <div className="text-right">
                    <div className="text-xs text-slate-400 mb-1">Monte Carlo</div>
                    <div className="text-sm font-semibold text-blue-400">
                      {visualizationPoints.length} samples
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* 3D Canvas */}
            <div className="w-full h-full">
              {visualizationPoints.length > 0 ? (
                <Canvas
                  camera={{ position: [3, 3, 5], fov: 50 }}
                  gl={{ antialias: true, alpha: true }}
                >
                  <color attach="background" args={["#0f172a"]} />
                  <ambientLight intensity={0.5} />
                  <pointLight position={[10, 10, 10]} intensity={1} />
                  <Suspense fallback={null}>
                    <RiskCloud
                      points={visualizationPoints}
                      targetX={suggestedTarget.x}
                      targetZ={suggestedTarget.z}
                      ghostPoints={ghostPoints.length > 0 ? ghostPoints : undefined}
                    />
                    {/* StrikeZone removed - RiskCloud already includes it */}
                    <Grid
                      args={[10, 10]}
                      cellSize={0.5}
                      cellColor="#1e293b"
                      sectionSize={1}
                      sectionColor="#334155"
                      fadeDistance={20}
                      fadeStrength={1}
                      position={[0, 0, 0]}
                    />
                  </Suspense>
                  <OrbitControls
                    enableZoom={true}
                    enablePan={true}
                    minDistance={2}
                    maxDistance={15}
                  />
                </Canvas>
              ) : (
                <div className="flex items-center justify-center h-full">
                  <div className="text-center text-slate-500">
                    <Loader2 className="w-8 h-8 animate-spin mx-auto mb-2" />
                    <p className="text-sm">Waiting for pitch selection...</p>
                  </div>
                </div>
              )}
            </div>

            {/* Footer */}
            <div className="absolute bottom-0 left-0 right-0 p-3 bg-slate-800/90 backdrop-blur-sm text-xs text-slate-400 text-center border-t border-slate-700">
              <div className="flex items-center justify-center gap-4">
                <span className="flex items-center gap-1">
                  <span className="inline-block w-2 h-2 rounded-full bg-cyan-400"></span>
                  Strike Zone
                </span>
                <span className="flex items-center gap-1">
                  <span className="inline-block w-2 h-2 rounded-full bg-yellow-400"></span>
                  Target Aim Point
                </span>
                <span className="flex items-center gap-1">
                  <span className="inline-block w-2 h-2 rounded-full bg-blue-400"></span>
                  Primary Pitch
                </span>
                {ghostPitch && (
                  <span className="flex items-center gap-1">
                    <span className="inline-block w-2 h-2 rounded-full bg-purple-400 opacity-40"></span>
                    Ghost (Comparison)
                  </span>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
