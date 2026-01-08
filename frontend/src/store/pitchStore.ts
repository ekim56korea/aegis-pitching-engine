import { create } from 'zustand';
import axios from 'axios';

// Types
interface PitchSpec {
  velocity: number;
  spinRate: number;
  spinAxis: number;
  extension: number;
  targetX: number;
  targetZ: number;
  releaseHeight?: number;
}

interface RiskPoint {
  x: number;
  y: number;
  z: number;
  isStrike?: boolean;
}

interface RiskMetrics {
  var95?: number;
  strikeRate: number;
  expectedValue?: number;
  variance?: number;
}

interface RiskData {
  points: RiskPoint[];
  metrics: RiskMetrics;
}

// Commander Logic Types
interface RankedPitch {
  pitch_id: string;
  pitch_type: string;
  zone: string;
  rank: number;
  xrv: number;
  var_95: number;
  strike_rate: number;
  intelligence_score: number;
  strategy_type: string;
  strategy_rationale: string;
  usage_percentage?: number;  // Optional: GTO-based recommended usage % (0-100)
  data_quality: number;
  is_reliable: boolean;
  scatter_sample?: Array<{ x: number; y: number; z: number; xrv: number }>;
  suggested_target?: [number, number];  // [x, z] AI-suggested target location
}

interface RecommendationResponse {
  game_context: {
    balls: number;
    strikes: number;
    batter_hand: string;
    pitcher_hand: string;
  };
  recommendations: RankedPitch[];
  best_pitch: string;
  scenario_description: string | null;
  has_warnings: boolean;
  warnings: string[];
}

interface PitchState {
  // Pitch Specification
  pitchSpec: PitchSpec;
  setPitchSpec: (spec: Partial<PitchSpec>) => void;

  // Risk Simulation
  riskData: RiskData | null;
  isRiskLoading: boolean;
  riskError: string | null;
  runRiskSimulation: () => Promise<void>;
  clearRiskData: () => void;

  // Trajectory Data (optional for future)
  trajectoryData: any | null;
  setTrajectoryData: (data: any) => void;

  // Commander Logic - Pitch Recommendations
  recommendations: RankedPitch[];
  bestPitchId: string | null;
  scenarioDescription: string | null;
  isRecommendationLoading: boolean;
  recommendationError: string | null;
  fetchRecommendations: (
    gameContext: { balls: number; strikes: number; batter_hand: 'L' | 'R'; pitcher_hand: 'L' | 'R' },
    pitches: Array<{
      pitch_type: string;
      velocity: number;
      spin_rate: number;
      spin_efficiency: number;
      spin_direction: number;
      release_height: number;
      release_side: number;
      extension: number;
      target_x: number | null;
      target_z: number | null;
    }>
  ) => Promise<void>;
  clearRecommendations: () => void;

  // Comparison Mode (Tunneling Visualization)
  comparePitchId: string | null;
  setComparePitchId: (id: string | null) => void;
  toggleCompareMode: (id: string) => void;
  getComparePitch: () => RankedPitch | undefined;
}

export const usePitchStore = create<PitchState>((set, get) => ({
  // Initial State
  pitchSpec: {
    velocity: 95,
    spinRate: 2400,
    spinAxis: 180,
    extension: 6.0,
    targetX: 0.0,
    targetZ: 2.5,
    releaseHeight: 6.0,
  },

  setPitchSpec: (spec) =>
    set((state) => ({
      pitchSpec: { ...state.pitchSpec, ...spec },
    })),

  // Risk Simulation State
  riskData: null,
  isRiskLoading: false,
  riskError: null,

  // Commander Logic State
  recommendations: [],
  bestPitchId: null,
  scenarioDescription: null,
  isRecommendationLoading: false,
  recommendationError: null,

  // Comparison Mode State
  comparePitchId: null,

  runRiskSimulation: async () => {
    const { pitchSpec } = get();

    set({ isRiskLoading: true, riskError: null });

    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000';

      // Build request payload matching RiskSimulationRequest schema
      const requestPayload = {
        pitch: {
          velocity: pitchSpec.velocity,
          spin_rate: pitchSpec.spinRate,
          spin_efficiency: 0.95,
          spin_direction: pitchSpec.spinAxis, // Required field: clock face degrees (0-360)
          axis_tilt: 0.0, // SSW axis tilt (optional)
          release_height: pitchSpec.releaseHeight || 6.0,
          release_side: -2.0,
          extension: pitchSpec.extension,
        },
        command_level: "average" as const,
        num_simulations: 1000,
      };

      console.log('🚀 Sending Risk Simulation Request:', JSON.stringify(requestPayload, null, 2));

      // API Request
      const response = await axios.post(`${apiUrl}/v1/simulation/risk`, requestPayload);

      console.log('📦 API Response:', response.data);

      const responseData = response.data;

      // Backend returns: { pitch_spec, command_level, num_simulations, risk_metrics, risk_classification, scatter_data }
      if (responseData && responseData.scatter_data) {
        // Sample first 5 points for debugging
        console.log('🔍 Sample Points (first 5):', responseData.scatter_data.slice(0, 5));

        // Get target coordinates from pitch spec
        const { targetX, targetZ } = pitchSpec;
        console.log(`🎯 Target: (${targetX.toFixed(2)}, ${targetZ.toFixed(2)})`);

        // Transform scatter_data to RiskPoint format
        // Backend returns DEVIATIONS from target, so add target coordinates to get absolute positions
        const points: RiskPoint[] = responseData.scatter_data.map((point: any) => {
          const x = (point.x || 0) + targetX;  // Backend deviation + target = absolute position
          const z = (point.z || 0) + targetZ;  // Backend deviation + target = absolute height

          // Check if point is in strike zone (MLB: -0.71 to 0.71 ft horizontal, 1.6 to 3.4 ft vertical)
          const isStrike = x >= -0.71 && x <= 0.71 && z >= 1.6 && z <= 3.4;

          return {
            x,
            y: 0, // At plate (home plate is y=0 in physics coordinates)
            z,
            isStrike,
          };
        });

        // Debug: Show range of coordinates AFTER adding target offset
        const xValues = points.map(p => p.x);
        const zValues = points.map(p => p.z);
        console.log('📊 Absolute Coordinate Ranges (after adding target):', {
          x: { min: Math.min(...xValues).toFixed(2), max: Math.max(...xValues).toFixed(2) },
          z: { min: Math.min(...zValues).toFixed(2), max: Math.max(...zValues).toFixed(2) },
          strikeZone: { x: '[-0.71, 0.71]', z: '[1.6, 3.4]' }
        });

        // Calculate strike rate
        const strikeCount = points.filter((p) => p.isStrike).length;
        const strikeRate = points.length > 0 ? (strikeCount / points.length) * 100 : 0;

        const riskData: RiskData = {
          points,
          metrics: {
            strikeRate,
            var95: responseData.risk_metrics?.var_95 || 0,
            expectedValue: responseData.risk_metrics?.mean_xrv || 0,
            variance: responseData.risk_metrics?.std_xrv || 0,
          },
        };

        console.log(`✅ Risk Simulation Complete: ${points.length} points, ${strikeRate.toFixed(1)}% strikes`);
        set({ riskData, isRiskLoading: false });
      } else {
        throw new Error('Invalid response format: missing scatter_data');
      }
    } catch (err: any) {
      console.group("🔥 Simulation Error Diagnostics");

      if (err.response) {
        // Server responded with error status code (2xx 범위 벗어남)
        console.error(`❌ HTTP Status: ${err.response.status}`);
        console.error("❌ Response Data:", err.response.data); // 핵심: 서버가 보낸 진짜 에러 메시지
        console.error("❌ Response Headers:", err.response.headers);
      } else if (err.request) {
        // 요청 전송됨, 그러나 응답 없음 (Network/CORS/Server Down)
        console.error("❌ No Response Received. Is the Backend running?");
        console.error("Request:", err.request);
      } else {
        // 요청 설정 중 에러
        console.error("❌ Request Setup Error:", err.message);
      }

      console.groupEnd();

      // Fallback: Generate mock data if API fails
      const mockPoints: RiskPoint[] = [];
      const { targetX, targetZ } = pitchSpec;

      for (let i = 0; i < 1000; i++) {
        // Box-Muller transform for normal distribution
        const u1 = Math.random();
        const u2 = Math.random();
        const z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
        const z1 = Math.sqrt(-2.0 * Math.log(u1)) * Math.sin(2.0 * Math.PI * u2);

        const x = targetX + z0 * 0.15;
        const z = targetZ + z1 * 0.12;

        const isStrike = x >= -0.71 && x <= 0.71 && z >= 1.6 && z <= 3.4;

        mockPoints.push({ x, y: 0, z, isStrike });
      }

      const strikeCount = mockPoints.filter((p) => p.isStrike).length;
      const strikeRate = (strikeCount / 1000) * 100;

      set({
        riskData: {
          points: mockPoints,
          metrics: { strikeRate },
        },
        isRiskLoading: false,
        riskError: err.response?.data?.detail
          ? JSON.stringify(err.response.data.detail)
          : "Simulation failed. Check console for details.",
      });
    }
  },

  clearRiskData: () => set({ riskData: null, riskError: null }),

  // Commander Logic - Fetch Recommendations
  fetchRecommendations: async (gameContext, pitches) => {
    set({ isRecommendationLoading: true, recommendationError: null });

    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000';

      const requestPayload = {
        context: {
          balls: gameContext.balls,
          strikes: gameContext.strikes,
          outs: 0,
          batter_hand: gameContext.batter_hand,
          pitcher_hand: gameContext.pitcher_hand,
          base_state: [false, false, false],
          inning: 5,
          score_diff: 0,
        },
        pitches: pitches.map((p, idx) => ({
          pitch_id: `pitch_${idx + 1}`,
          pitch_type: p.pitch_type,
          zone: "middle",
          pitch_spec: {
            velocity: p.velocity,
            spin_rate: p.spin_rate,
            spin_efficiency: p.spin_efficiency,
            spin_direction: p.spin_direction,
            release_height: p.release_height,
            release_side: p.release_side,
            extension: p.extension,
          },
          target_location: {
            x: p.target_x,
            z: p.target_z,
          },
        })),
        num_simulations: 1000,
      };

      console.log('🧠 Fetching Recommendations:', JSON.stringify(requestPayload, null, 2));

      const response = await axios.post<RecommendationResponse>(
        `${apiUrl}/v1/simulation/recommend`,
        requestPayload
      );

      console.log('✅ Recommendations received:', response.data);

      set({
        recommendations: response.data.recommendations,
        bestPitchId: response.data.best_pitch,
        scenarioDescription: response.data.scenario_description,
        isRecommendationLoading: false,
      });
    } catch (err: any) {
      console.error('❌ Recommendation Error:', err);
      set({
        recommendationError: err.response?.data?.detail || 'Failed to fetch recommendations',
        isRecommendationLoading: false,
      });
    }
  },

  clearRecommendations: () =>
    set({
      recommendations: [],
      bestPitchId: null,
      scenarioDescription: null,
      recommendationError: null,
    }),

  // Comparison Mode Actions
  setComparePitchId: (id) => set({ comparePitchId: id }),

  toggleCompareMode: (id) => {
    const { comparePitchId, bestPitchId } = get();

    // If this pitch is already the comparison, clear it
    if (comparePitchId === id) {
      set({ comparePitchId: null });
      return;
    }

    // If this pitch is the primary (best), do nothing (can't compare to itself)
    if (bestPitchId === id) {
      console.warn('Cannot compare primary pitch to itself');
      return;
    }

    // Otherwise, set as comparison pitch
    set({ comparePitchId: id });
  },

  getComparePitch: () => {
    const { comparePitchId, recommendations } = get();
    if (!comparePitchId) return undefined;
    return recommendations.find((pitch) => pitch.pitch_id === comparePitchId);
  },

  // Trajectory Data (for future trajectory API integration)
  trajectoryData: null,
  setTrajectoryData: (data) => set({ trajectoryData: data }),
}));
