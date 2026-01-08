"""
Batch simulation service for high-performance pitch comparison
"""
import time
import logging
import numpy as np
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.api.schemas import (
    PitchSetup,
    PitchAnalysis,
    RiskMetrics,
    ScatterPoint,
    GameContext
)
from src.simulation.monte_carlo import ScenarioSimulator

logger = logging.getLogger(__name__)


class BatchSimulationService:
    """
    High-performance batch simulation service

    Optimizations:
    - Thread pool for parallel pitch simulations
    - Efficient scatter data sampling (max 200 points)
    - Context-adjusted metric calculations

    Target: <500ms for 4 pitches × 1000 simulations
    """

    def __init__(self, simulator: ScenarioSimulator, max_workers: int = 4):
        """
        Initialize batch service

        Args:
            simulator: Monte Carlo simulator instance
            max_workers: Thread pool size for parallel execution
        """
        self.simulator = simulator
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def simulate_batch(
        self,
        pitches: List[PitchSetup],
        game_context: GameContext,
        command_level: str,
        num_simulations: int
    ) -> Tuple[List[PitchAnalysis], float, np.ndarray]:
        """
        Run batch Monte Carlo simulation for multiple pitches

        Args:
            pitches: List of pitch configurations
            game_context: Shared game situation
            command_level: Command quality
            num_simulations: Monte Carlo trials per pitch

        Returns:
            analyses: List of PitchAnalysis with context metrics
            total_time: Total computation time
            cov_matrix: Covariance matrix used
        """
        start_time = time.perf_counter()

        # Get covariance matrix for reference
        profile = self.simulator.command_profiles.get(command_level)
        if profile is None:
            raise ValueError(f"Unknown command level: {command_level}")
        cov_matrix = self.simulator._build_covariance_matrix(profile)

        logger.info(f"Batch simulation: {len(pitches)} pitches, {num_simulations} sims each")

        # Submit all simulations to thread pool
        future_to_pitch = {}
        for pitch_setup in pitches:
            future = self.executor.submit(
                self._simulate_single_pitch,
                pitch_setup,
                command_level,
                num_simulations
            )
            future_to_pitch[future] = pitch_setup

        # Collect results as they complete
        analyses = []
        for future in as_completed(future_to_pitch):
            pitch_setup = future_to_pitch[future]
            try:
                report = future.result()

                # Build analysis with context adjustments
                analysis = self._build_pitch_analysis(
                    pitch_setup,
                    report,
                    game_context
                )
                analyses.append(analysis)

            except Exception as e:
                logger.error(f"Simulation failed for {pitch_setup.pitch_id}: {e}")
                raise

        # Sort analyses by original pitch order
        pitch_id_order = {p.pitch_id: i for i, p in enumerate(pitches)}
        analyses.sort(key=lambda a: pitch_id_order[a.pitch_id])

        total_time = time.perf_counter() - start_time
        logger.info(f"Batch complete: {total_time:.3f}s ({total_time/len(pitches)*1000:.1f}ms per pitch)")

        return analyses, total_time, cov_matrix

    def _simulate_single_pitch(
        self,
        pitch_setup: PitchSetup,
        command_level: str,
        num_simulations: int
    ):
        """Run simulation for a single pitch"""
        pitch_dict = pitch_setup.spec.model_dump()

        # Override num_simulations temporarily
        original_num_sims = self.simulator.config['num_simulations']
        self.simulator.config['num_simulations'] = num_simulations

        try:
            report = self.simulator.simulate_risk(pitch_dict, command_level)
            # Attach target to report for later use in strike probability calculation
            report.target = pitch_setup.target
            return report
        finally:
            self.simulator.config['num_simulations'] = original_num_sims

    def _build_pitch_analysis(
        self,
        pitch_setup: PitchSetup,
        report,
        game_context: GameContext
    ) -> PitchAnalysis:
        """Build PitchAnalysis with context-adjusted metrics"""

        # Extract raw metrics
        risk_metrics = RiskMetrics(
            mean_xrv=report.mean_xrv,
            std_xrv=report.std_xrv,
            median_xrv=report.median_xrv,
            var_95=report.var_95,
            sharpe_ratio=report.sharpe_ratio,
            percentiles=report.xrv_percentiles,
            barrel_rate=report.barrel_rate
        )

        # Calculate context adjustments
        context_adjusted_xrv = report.mean_xrv * game_context.leverage_index

        # Calculate count transition deltas
        strike_delta = game_context.calculate_outcome_delta("strike")
        ball_delta = game_context.calculate_outcome_delta("ball")

        # Estimate strike probability
        strike_prob = self._estimate_strike_probability(report)

        # Expected count delta (probability-weighted)
        expected_delta = (strike_prob * strike_delta) + ((1 - strike_prob) * ball_delta)

        # Total value = context xRV + expected count delta
        total_value = context_adjusted_xrv + expected_delta

        # Sample scatter data (max 200 points)
        scatter_sample = self._sample_scatter_data(report)

        return PitchAnalysis(
            pitch_id=pitch_setup.pitch_id,
            pitch_type=pitch_setup.pitch_type,
            pitch_spec=pitch_setup.spec.model_dump(),
            risk_metrics=risk_metrics,
            risk_classification=report.risk_category,
            context_adjusted_xrv=context_adjusted_xrv,
            expected_strike_delta=strike_delta,
            expected_ball_delta=ball_delta,
            strike_probability=strike_prob,
            total_value=total_value,
            scatter_sample=scatter_sample
        )

    def _sample_scatter_data(self, report) -> List[ScatterPoint]:
        """Sample scatter data for visualization"""
        viz_data = report.visualization_data
        x_array = viz_data.get('x', [])  # Noise/Deviation from target (meters)
        z_array = viz_data.get('z', [])  # Noise/Deviation from target (meters)
        xrv_array = viz_data.get('xrv', [])

        logger.info(f"Scatter sampling: x type={type(x_array)}, len={len(x_array) if hasattr(x_array, '__len__') else 'N/A'}")

        if not isinstance(x_array, np.ndarray) or len(x_array) == 0:
            logger.warning("Empty or invalid scatter data")
            return []

        # [CRITICAL FIX] Get target location (default to strike zone center if not provided)
        # Physics engine returns DEVIATION from target, so we must add target coordinates
        # to get ABSOLUTE landing positions
        target = getattr(report, 'target', None)
        if target is None:
            target_x_ft, target_z_ft = 0.0, 2.5  # Strike zone center (feet)
        else:
            target_x_ft, target_z_ft = target[0], target[1]

        num_points = len(x_array)
        sample_size = min(200, num_points)
        sample_indices = np.random.choice(num_points, size=sample_size, replace=False)

        # Meters to Feet Conversion Constant
        M_TO_FT = 3.28084

        points = [
            ScatterPoint(
                # Convert deviation (meters) to feet, then add target position (feet)
                x=(float(x_array[i]) * M_TO_FT) + target_x_ft,
                z=(float(z_array[i]) * M_TO_FT) + target_z_ft,
                xrv=float(xrv_array[i])
            )
            for i in sample_indices
        ]

        logger.info(f"Sampled {len(points)} scatter points from {num_points} (target: {target_x_ft:.2f}, {target_z_ft:.2f})")
        return points

    def _estimate_strike_probability(self, report) -> float:
        """Estimate strike probability from scatter data"""
        viz_data = report.visualization_data
        x_noise = viz_data.get('x', np.array([]))  # Deviation from target
        z_noise = viz_data.get('z', np.array([]))  # Deviation from target

        # Meters to Feet Conversion (Critical for correct strike zone calculation)
        M_TO_FT = 3.28084
        x_noise = x_noise * M_TO_FT
        z_noise = z_noise * M_TO_FT

        if len(x_noise) == 0:
            return 0.0

        # Get target location (default to strike zone center if not provided)
        target = getattr(report, 'target', None)
        if target is None:
            target_x, target_z = 0.0, 2.5  # Strike zone center
        else:
            target_x, target_z = target[0], target[1]

        # Calculate absolute landing positions (target + noise)
        x_coords = x_noise + target_x
        z_coords = z_noise + target_z

        # MLB strike zone boundaries (in feet)
        x_min, x_max = -0.71, 0.71
        z_min, z_max = 1.6, 3.4

        in_zone = (
            (x_coords >= x_min) & (x_coords <= x_max) &
            (z_coords >= z_min) & (z_coords <= z_max)
        )

        return float(np.mean(in_zone))

    def rank_recommendations(self, analyses: List[PitchAnalysis]) -> List[dict]:
        """Rank pitches by total_value (best first)"""
        ranked = sorted(analyses, key=lambda a: a.total_value, reverse=True)

        return [
            {
                'pitch_id': a.pitch_id,
                'pitch_type': a.pitch_type,
                'total_value': a.total_value,
                'context_adjusted_xrv': a.context_adjusted_xrv,
                'strike_probability': a.strike_probability,
                'risk_label': a.risk_classification
            }
            for a in ranked
        ]
