"use client";

import { useRef, useMemo, useLayoutEffect } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";

interface RiskPoint {
  x: number;
  y: number;
  z: number;
  isStrike?: boolean;
  xRV?: number;
}

interface RiskCloudProps {
  points: RiskPoint[];
  targetX?: number;
  targetZ?: number;
  ghostPoints?: RiskPoint[];  // Optional: Secondary pitch for tunneling visualization
}

// ============================================================================
// COORDINATE TRANSFORMATION PIPELINE
// ============================================================================
// Physics Engine: Z-up coordinate system (meters/feet)
//   - X: Horizontal position (feet, + = right from pitcher's view)
//   - Y: Distance from pitcher (feet, 0 = home plate ~60.5ft from rubber)
//   - Z: Height above ground (feet)
//
// Three.js: Y-up coordinate system (arbitrary units)
//   - X: Horizontal position (+ = right from camera view)
//   - Y: Vertical position (height)
//   - Z: Depth (+ = away from camera)
//
// Viewing Perspective: Catcher's View (looking from behind home plate)
// ============================================================================

/**
 * Scale factor for depth mapping. Adjusts how compressed/stretched the
 * depth dimension appears. Higher values = more compressed (flatter view).
 */
const DEPTH_SCALE_FACTOR = 0.05;

/**
 * Home plate distance from pitcher's rubber (feet).
 * Physics engine uses this as y=0 reference point.
 */
const HOME_PLATE_DISTANCE = 60.5;

/**
 * Transform physics coordinates to Three.js graphics coordinates.
 *
 * @param point Physics engine point {x, y, z}
 * @returns Three.js position {x, y, z}
 */
function transformCoordinates(point: { x: number; y: number; z: number }): {
  x: number;
  y: number;
  z: number;
} {
  return {
    // X (Horizontal): Invert for Catcher's View
    // RHP throws from left side of screen, LHP from right
    x: point.x * -1,

    // Y (Vertical/Height): Physics Z-height maps to Graphics Y-up
    y: point.z,

    // Z (Depth/Distance): Map plate distance to screen depth
    // Negative scale moves points away from camera as they approach plate
    z: (point.y - HOME_PLATE_DISTANCE) * DEPTH_SCALE_FACTOR
  };
}

export default function RiskCloud({
  points,
  targetX = 0,
  targetZ = 2.5,
  ghostPoints
}: RiskCloudProps) {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const ghostMeshRef = useRef<THREE.InstancedMesh>(null);
  const tempObject = useMemo(() => new THREE.Object3D(), []);

  // Strike zone bounds (standard MLB, in feet)
  // Width: 17 inches = 1.417 feet (±0.708 from center)
  // Height: Typically 1.5ft (knees) to 3.5ft (letters) above ground
  const strikeZone = {
    xMin: -0.708,
    xMax: 0.708,
    zMin: 1.5,
    zMax: 3.5
  };

  // Color palette for distance gradient and dirt balls
  const colorSafe = useMemo(() => new THREE.Color("#3b82f6"), []); // Blue (safe)
  const colorRisk = useMemo(() => new THREE.Color("#ef4444"), []); // Red (risk)
  const colorDirt = useMemo(() => new THREE.Color("#78350f"), []); // Brown (dirt ball)

  // Calculate max distance for normalization
  const maxDistance = useMemo(() => {
    let max = 0;
    points.forEach((point) => {
      const dx = point.x - targetX;
      const dz = point.z - targetZ;
      const dist = Math.sqrt(dx * dx + dz * dz);
      if (dist > max) max = dist;
    });
    return max || 1;
  }, [points, targetX, targetZ]);

  // High-performance matrix & color setup with useLayoutEffect
  useLayoutEffect(() => {
    if (!meshRef.current) return;

    const colors: number[] = [];

    points.forEach((point, i) => {
      // ========================================================================
      // APPLY COORDINATE TRANSFORMATION PIPELINE
      // ========================================================================
      const transformed = transformCoordinates(point);

      tempObject.position.set(transformed.x, transformed.y, transformed.z);

      // Check if ball hit the ground (dirt ball)
      const isDirtBall = point.z < 0;

      // For dirt balls, flatten the sphere to indicate ground contact
      if (isDirtBall) {
        tempObject.scale.set(1.5, 0.3, 1.5); // Flattened
      } else {
        tempObject.scale.set(1, 1, 1); // Normal
      }

      tempObject.updateMatrix();
      meshRef.current!.setMatrixAt(i, tempObject.matrix);

      // ========================================================================
      // COLOR ASSIGNMENT: Distance Gradient or Dirt Ball
      // ========================================================================
      let color: THREE.Color;

      if (isDirtBall) {
        // Dirt balls get dark brown color with reduced opacity
        color = colorDirt.clone();
      } else {
        // Distance-based gradient from target
        const dx = point.x - targetX;
        const dz = point.z - targetZ;
        const distance = Math.sqrt(dx * dx + dz * dz);
        const normalizedDistance = Math.min(distance / maxDistance, 1);

        // Interpolate between safe (blue) and risk (red)
        color = colorSafe.clone().lerp(colorRisk, normalizedDistance);
      }

      colors.push(color.r, color.g, color.b);
    });

    meshRef.current.instanceMatrix.needsUpdate = true;

    // Set vertex colors
    const colorAttribute = new THREE.InstancedBufferAttribute(
      new Float32Array(colors),
      3
    );
    meshRef.current.geometry.setAttribute("color", colorAttribute);
  }, [points, targetX, targetZ, maxDistance, colorSafe, colorRisk, colorDirt, tempObject]);

  // ============================================================================
  // GHOST CLOUD SETUP (Tunneling Visualization)
  // ============================================================================
  useLayoutEffect(() => {
    if (!ghostMeshRef.current || !ghostPoints || ghostPoints.length === 0) return;

    ghostPoints.forEach((point, i) => {
      // Apply same coordinate transformation as primary pitch
      const transformed = transformCoordinates(point);

      tempObject.position.set(transformed.x, transformed.y, transformed.z);

      // Check if ball hit the ground
      const isDirtBall = point.z < 0;

      // For dirt balls, flatten the sphere
      if (isDirtBall) {
        tempObject.scale.set(1.5, 0.3, 1.5);
      } else {
        tempObject.scale.set(1, 1, 1);
      }

      tempObject.updateMatrix();
      ghostMeshRef.current!.setMatrixAt(i, tempObject.matrix);
    });

    ghostMeshRef.current.instanceMatrix.needsUpdate = true;
  }, [ghostPoints, tempObject]);

  // Subtle floating animation for visual appeal
  useFrame((state) => {
    if (meshRef.current) {
      meshRef.current.rotation.y = Math.sin(state.clock.elapsedTime * 0.3) * 0.03;
    }
    // Ghost cloud follows same animation for visual coherence
    if (ghostMeshRef.current) {
      ghostMeshRef.current.rotation.y = Math.sin(state.clock.elapsedTime * 0.3) * 0.03;
    }
  });

  if (points.length === 0) return null;

  // ============================================================================
  // STRIKE ZONE TRANSFORMED COORDINATES
  // ============================================================================
  // Transform strike zone bounds to graphics coordinates
  const strikeZoneGraphics = {
    // Horizontal bounds (inverted for catcher's view)
    xMin: strikeZone.xMin * -1,
    xMax: strikeZone.xMax * -1,
    // Vertical bounds (direct mapping)
    yMin: strikeZone.zMin,
    yMax: strikeZone.zMax,
    // Depth at home plate (y=0 in physics)
    z: (0 - HOME_PLATE_DISTANCE) * DEPTH_SCALE_FACTOR
  };

  const strikeZoneWidth = Math.abs(strikeZoneGraphics.xMax - strikeZoneGraphics.xMin);
  const strikeZoneHeight = strikeZoneGraphics.yMax - strikeZoneGraphics.yMin;
  const strikeZoneCenterX = (strikeZoneGraphics.xMin + strikeZoneGraphics.xMax) / 2;
  const strikeZoneCenterY = (strikeZoneGraphics.yMin + strikeZoneGraphics.yMax) / 2;

  return (
    <group>
      {/* Strike Zone Wireframe Box */}
      <lineSegments
        position={[strikeZoneCenterX, strikeZoneCenterY, strikeZoneGraphics.z]}
      >
        <edgesGeometry
          args={[
            new THREE.BoxGeometry(
              strikeZoneWidth,
              strikeZoneHeight,
              0.01
            )
          ]}
        />
        <lineBasicMaterial
          color="#10b981"
          opacity={0.6}
          transparent
          linewidth={2}
        />
      </lineSegments>

      {/* Point Cloud */}
      <instancedMesh ref={meshRef} args={[undefined, undefined, points.length]}>
        <sphereGeometry args={[0.05, 8, 8]} />
        <meshStandardMaterial
          transparent
          opacity={0.6}
          depthWrite={false}
          vertexColors
          emissive="#ffffff"
          emissiveIntensity={0.2}
          roughness={0.3}
          metalness={0.1}
        />
      </instancedMesh>

      {/* Ghost Point Cloud (Tunneling Comparison) */}
      {ghostPoints && ghostPoints.length > 0 && (
        <instancedMesh ref={ghostMeshRef} args={[undefined, undefined, ghostPoints.length]}>
          <sphereGeometry args={[0.05, 8, 8]} />
          <meshStandardMaterial
            color="#e2e8f0"
            transparent
            opacity={0.15}
            depthWrite={false}
            roughness={0.4}
            metalness={0.2}
          />
        </instancedMesh>
      )}

      {/* Target Aim Point Marker */}
      <group
        position={[
          transformCoordinates({ x: targetX, y: 0, z: targetZ }).x,
          transformCoordinates({ x: targetX, y: 0, z: targetZ }).y,
          transformCoordinates({ x: targetX, y: 0, z: targetZ }).z
        ]}
      >
        {/* Outer glow ring */}
        <mesh>
          <ringGeometry args={[0.08, 0.12, 32]} />
          <meshBasicMaterial
            color="#fbbf24"
            transparent
            opacity={0.4}
            side={THREE.DoubleSide}
          />
        </mesh>

        {/* Inner crosshair */}
        <mesh>
          <ringGeometry args={[0.03, 0.05, 32]} />
          <meshBasicMaterial
            color="#fbbf24"
            transparent
            opacity={0.8}
            side={THREE.DoubleSide}
          />
        </mesh>

        {/* Center dot */}
        <mesh>
          <sphereGeometry args={[0.04, 16, 16]} />
          <meshStandardMaterial
            color="#fbbf24"
            emissive="#fbbf24"
            emissiveIntensity={0.8}
            transparent
            opacity={0.9}
          />
        </mesh>

        {/* Vertical line */}
        <mesh position={[0, 0.08, 0]}>
          <boxGeometry args={[0.01, 0.1, 0.01]} />
          <meshBasicMaterial color="#fbbf24" transparent opacity={0.6} />
        </mesh>
        <mesh position={[0, -0.08, 0]}>
          <boxGeometry args={[0.01, 0.1, 0.01]} />
          <meshBasicMaterial color="#fbbf24" transparent opacity={0.6} />
        </mesh>

        {/* Horizontal line */}
        <mesh position={[0.08, 0, 0]}>
          <boxGeometry args={[0.1, 0.01, 0.01]} />
          <meshBasicMaterial color="#fbbf24" transparent opacity={0.6} />
        </mesh>
        <mesh position={[-0.08, 0, 0]}>
          <boxGeometry args={[0.1, 0.01, 0.01]} />
          <meshBasicMaterial color="#fbbf24" transparent opacity={0.6} />
        </mesh>
      </group>
    </group>
  );
}
