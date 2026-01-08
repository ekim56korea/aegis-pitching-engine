"""
Pitch Recommendation API - Commander Logic Integration

Provides intelligent pitch selection recommendations using batch simulation
and count-adaptive Commander Logic scoring.
"""

from fastapi import APIRouter, Depends, HTTPException, Request
from typing import List, Optional
import logging
import time

from src.api.schemas import (
    GameContext,
    PitchSetup,
    PitchAnalysis,
)
from src.api.batch_simulation import BatchSimulationService
from src.services.recommendation import PitchRecommendationService
from src.simulation import ScenarioSimulator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/simulation", tags=["recommendation"])


# Pydantic models for recommendation API
from pydantic import BaseModel, Field


class PitchConfig(BaseModel):
    """Single pitch configuration for recommendation"""
    pitch_id: str
    pitch_type: str
    zone: str
    pitch_spec: dict
    target_location: dict


class RecommendationRequest(BaseModel):
    """Request for pitch recommendations"""
    context: GameContext
    pitches: List[PitchConfig] = Field(..., min_items=1, max_items=10)
    num_simulations: int = Field(default=1000, ge=100, le=5000)


class RankedPitch(BaseModel):
    """Ranked pitch recommendation with Commander Logic scoring"""
    pitch_id: str
    pitch_type: str
    zone: str
    rank: int
    xrv: float
    var_95: float
    strike_rate: float
    intelligence_score: float
    usage_percentage: float = Field(default=0.0, description="GTO-based recommended usage % (Softmax)")
    strategy_type: str
    strategy_rationale: str
    data_quality: float
    is_reliable: bool
    scatter_sample: List[dict] = Field(default_factory=list, description="3D scatter points for visualization")
    suggested_target: Optional[List[float]] = Field(default=None, description="AI-suggested target location [x, z] in feet")


class RecommendationResponse(BaseModel):
    """Recommendation response with ranked pitches"""
    game_context: GameContext
    recommendations: List[RankedPitch]
    best_pitch: str
    scenario_description: Optional[str] = None
    has_warnings: bool = False
    warnings: List[str] = []


def get_batch_service(request: Request) -> BatchSimulationService:
    """Dependency: Get batch simulation service"""
    if not hasattr(request.app.state, 'simulator') or request.app.state.simulator is None:
        raise HTTPException(status_code=503, detail="Simulator not initialized")

    simulator = request.app.state.simulator
    return BatchSimulationService(simulator=simulator, max_workers=4)


def get_recommendation_service() -> PitchRecommendationService:
    """Dependency: Get recommendation service"""
    return PitchRecommendationService()


@router.post("/recommend", response_model=RecommendationResponse)
async def recommend_pitch(
    request: RecommendationRequest,
    batch_service: BatchSimulationService = Depends(get_batch_service),
    rec_service: PitchRecommendationService = Depends(get_recommendation_service)
):
    """
    Get intelligent pitch recommendations using Commander Logic

    This endpoint:
    1. Runs batch Monte Carlo simulation for all pitches
    2. Applies count-adaptive Commander Logic scoring
    3. Ranks pitches by intelligence score (lower is better)
    4. Returns ranked recommendations with rationale

    Args:
        request: Game context and pitch configurations

    Returns:
        RecommendationResponse with ranked pitches and strategy insights

    Performance:
        - Target: <100ms for 4 pitches × 1000 simulations
        - Physics: ~3-5ms per pitch (batch parallel)
        - Ranking: <5ms for Commander Logic scoring
    """
    start_time = time.time()

    logger.info(f"🧠 Recommendation Request: {request.context.balls}-{request.context.strikes} count, {len(request.pitches)} pitches")

    try:
        # Get count strategy for auto-targeting
        from src.services.recommendation import StrategyMatrix, TargetHeatmap
        strategy_weights = StrategyMatrix.get_weights(request.context.balls, request.context.strikes)

        # Step 1: Convert PitchConfig to PitchSetup for batch simulation
        pitch_setups = []
        for pitch_config in request.pitches:
            spec_data = pitch_config.pitch_spec
            target = pitch_config.target_location

            # Create PitchSpec from dict
            from src.api.schemas import PitchSpec

            pitch_spec = PitchSpec(
                velocity=spec_data.get('velocity', 95),
                spin_rate=spec_data.get('spin_rate', 2400),
                spin_efficiency=spec_data.get('spin_efficiency', 0.95),
                spin_direction=spec_data.get('spin_direction', 180),
                axis_tilt=spec_data.get('axis_tilt', 0.0),
                release_height=spec_data.get('release_height', 6.0),
                release_side=spec_data.get('release_side', -2.0),
                extension=spec_data.get('extension', 6.0),
                gyro_degree=0.0,  # Default value
                horizontal_break=0.0,  # Will be calculated
                induced_vertical_break=0.0  # Will be calculated
            )

            # AUTO-TARGETING: Assign optimal target if missing or incomplete
            # Check if target is explicitly provided and has both x and z coordinates
            has_valid_target = (
                target and
                isinstance(target, dict) and
                'x' in target and
                'z' in target and
                target.get('x') is not None and
                target.get('z') is not None
            )

            if has_valid_target:
                # Target explicitly provided by frontend
                target_loc = [target.get('x'), target.get('z')]
                logger.debug(f"✓ Using explicit target: {pitch_config.pitch_type} → ({target_loc[0]:.2f}, {target_loc[1]:.2f})")
            else:
                # AUTO-TARGETING: Assign optimal target based on pitch type and count strategy
                # For Sweepers (ST), pass pitch_spec for dynamic calculation
                pitch_spec_dict = {
                    'spin_direction': spec_data.get('spin_direction', 180),
                    'spin_efficiency': spec_data.get('spin_efficiency', 0.95),
                    'velocity': spec_data.get('velocity', 85)
                }

                optimal_x, optimal_z = TargetHeatmap.get_optimal_target(
                    pitch_type=pitch_config.pitch_type,
                    strategy=strategy_weights.strategy,
                    pitcher_hand=request.context.pitcher_hand.value,
                    batter_hand=request.context.batter_hand.value,
                    pitch_spec=pitch_spec_dict
                )
                target_loc = [optimal_x, optimal_z]

                # Log with physics details for ST (Dynamic Calculation)
                if pitch_config.pitch_type.upper() == 'ST':
                    logger.info(
                        f"🎯 Auto-targeting (DYNAMIC): {pitch_config.pitch_type} → "
                        f"({optimal_x:.2f}, {optimal_z:.2f}) [{strategy_weights.strategy.value}] "
                        f"[spin_dir={pitch_spec_dict['spin_direction']:.0f}°, eff={pitch_spec_dict['spin_efficiency']:.2f}]"
                    )
                else:
                    logger.info(
                        f"🎯 Auto-targeting (HEATMAP): {pitch_config.pitch_type} → "
                        f"({optimal_x:.2f}, {optimal_z:.2f}) [{strategy_weights.strategy.value}]"
                    )

            pitch_setup = PitchSetup(
                pitch_id=pitch_config.pitch_id,
                pitch_type=pitch_config.pitch_type,
                spec=pitch_spec,
                target=target_loc
            )
            pitch_setups.append(pitch_setup)

        # Create target lookup for later use in response
        target_lookup = {ps.pitch_id: ps.target for ps in pitch_setups}

        # Step 2: Run batch simulation (parallel physics)
        physics_start = time.time()
        analyses, batch_time, cov_matrix = batch_service.simulate_batch(
            pitches=pitch_setups,
            game_context=request.context,
            command_level="average",
            num_simulations=request.num_simulations
        )
        physics_time = (time.time() - physics_start) * 1000

        logger.info(f"   ⚙️  Physics simulation: {physics_time:.1f}ms for {len(analyses)} pitches")

        # Step 3: Apply Commander Logic ranking
        ranking_start = time.time()
        ranked_pitch_objects = rec_service.rank_pitches(
            pitch_analyses=analyses,
            balls=request.context.balls,
            strikes=request.context.strikes
        )
        ranking_time = (time.time() - ranking_start) * 1000

        logger.info(f"   🧠 Commander Logic ranking: {ranking_time:.1f}ms")

        # Step 4: Convert to response format
        ranked_pitches = []
        warnings = []

        for ranked_obj in ranked_pitch_objects:
            # Check data quality (based on simulation count)
            data_quality = min(1.0, request.num_simulations / 1000.0)
            is_reliable = data_quality >= 0.7

            if not is_reliable:
                warnings.append(
                    f"{ranked_obj.pitch_type}: Low simulation count (n={request.num_simulations})"
                )

            # Find corresponding analysis for scatter data and target
            analysis = next((a for a in analyses if a.pitch_id == ranked_obj.pitch_id), None)
            scatter_data = []
            if analysis and analysis.scatter_sample:
                scatter_data = [
                    {"x": pt.x, "y": 0.0, "z": pt.z, "xrv": pt.xrv}
                    for pt in analysis.scatter_sample
                ]

            # Get the target that was actually used
            suggested_target = target_lookup.get(ranked_obj.pitch_id)

            ranked_pitch = RankedPitch(
                pitch_id=ranked_obj.pitch_id,
                pitch_type=ranked_obj.pitch_type,
                zone="middle",  # Get from analysis if available
                rank=ranked_obj.rank,
                xrv=ranked_obj.xrv,
                var_95=ranked_obj.var_95,
                strike_rate=ranked_obj.strike_probability,
                intelligence_score=ranked_obj.score,
                usage_percentage=ranked_obj.usage_percentage,  # GTO-based usage %
                strategy_type=ranked_obj.strategy_used.value,
                strategy_rationale=ranked_obj.rationale,
                data_quality=data_quality,
                is_reliable=is_reliable,
                scatter_sample=scatter_data,
                suggested_target=suggested_target
            )
            ranked_pitches.append(ranked_pitch)

        # Step 5: Generate scenario description
        scenario_desc = _generate_scenario_description(request.context)

        total_time = (time.time() - start_time) * 1000
        logger.info(f"   ✅ Total recommendation time: {total_time:.1f}ms")

        return RecommendationResponse(
            game_context=request.context,
            recommendations=ranked_pitches,
            best_pitch=ranked_pitches[0].pitch_id if ranked_pitches else "",
            scenario_description=scenario_desc,
            has_warnings=len(warnings) > 0,
            warnings=warnings
        )

    except Exception as e:
        logger.error(f"❌ Recommendation error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Recommendation generation failed: {str(e)}"
        )


def _generate_scenario_description(context: GameContext) -> str:
    """Generate natural language description of game situation"""

    count_desc = f"{context.balls}-{context.strikes}"

    # Count pressure
    if context.strikes == 2:
        pressure = "two-strike pressure"
    elif context.balls == 3:
        pressure = "full count pressure" if context.strikes == 2 else "hitter's count"
    elif context.balls == 0 and context.strikes == 0:
        pressure = "fresh count"
    else:
        pressure = "neutral count"

    # Matchup - Extract enum values explicitly
    matchup = f"{context.batter_hand.value}HH vs {context.pitcher_hand.value}HP"

    return f"{count_desc} count ({pressure}), {matchup}"
