"""
Pitch Recommendation Service - Commander Logic Implementation

This module implements the "Commander Logic" scoring algorithm that merges:
- xRV (Pitch Performance/Stuff)
- VaR (Risk/Control)

Based on count-specific strategies to recommend optimal pitch selection.

Core Philosophy:
    - Aggressive counts (0-2, 1-2): Maximize strikeout potential (high xRV weight)
    - Defensive counts (3-0, 3-1, 3-2): Minimize walk risk (high VaR weight)
    - Neutral counts: Balanced approach

Score Interpretation:
    - Lower score is BETTER (negative xRV is good for pitcher)
    - Score = w_stuff × xRV + w_control × VaR
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Literal, Optional
from enum import Enum
import math


# ============================================================================
# SOFTMAX CONFIGURATION
# ============================================================================
# Temperature parameter for GTO usage percentage calculation
# - Lower T (0.1-0.3): Exploit mode (heavily favor top pick)
# - Default T (0.5): Balanced GTO mix
# - Higher T (0.7-1.0): Flatten distribution (more variety)
SOFTMAX_TEMPERATURE = 0.5

# Minimum strike probability threshold (pitches below this are non-competitive)
MIN_STRIKE_RATE_THRESHOLD = 0.10  # 10% - waste pitches excluded


class StrategyType(str, Enum):
    """Pitch selection strategy based on count"""
    AGGRESSIVE = "Aggressive"      # Attack mode - go for strikeout
    BALANCED = "Balanced"          # Neutral approach
    CONSERVATIVE = "Conservative"  # Defensive mode - avoid walks


@dataclass
class StrategyWeights:
    """
    Scoring weights for a specific count situation

    Attributes:
        w_stuff: Weight for xRV (pitch stuff/performance)
        w_control: Weight for VaR (command/risk)
        strategy: Strategy classification

    Note: w_stuff + w_control should equal 1.0
    """
    w_stuff: float      # Weight for xRV (performance)
    w_control: float    # Weight for VaR (risk/control)
    strategy: StrategyType

    def __post_init__(self):
        """Validate weights sum to 1.0"""
        total = self.w_stuff + self.w_control
        if not (0.99 <= total <= 1.01):  # Allow small floating point error
            raise ValueError(f"Weights must sum to 1.0, got {total}")


class StrategyMatrix:
    """
    Count-specific strategy matrix for pitch recommendation

    Maps each ball-strike count to optimal weighting strategy.

    Strategy Philosophy:
        - Pitcher's count (ahead): Aggressive - prioritize strikeout
        - Even count: Balanced - mix of stuff and control
        - Hitter's count (behind): Conservative - prioritize strikes
    """

    # Core strategy matrix: (balls, strikes) -> (w_stuff, w_control, strategy)
    MATRIX: Dict[Tuple[int, int], StrategyWeights] = {
        # Pitcher's counts - Aggressive (maximize whiff potential)
        (0, 2): StrategyWeights(0.80, 0.20, StrategyType.AGGRESSIVE),  # Two-strike: go for K
        (1, 2): StrategyWeights(0.75, 0.25, StrategyType.AGGRESSIVE),  # Still aggressive
        (2, 2): StrategyWeights(0.65, 0.35, StrategyType.AGGRESSIVE),  # Less aggressive

        # Neutral counts - Balanced
        (0, 0): StrategyWeights(0.50, 0.50, StrategyType.BALANCED),    # First pitch
        (1, 0): StrategyWeights(0.55, 0.45, StrategyType.BALANCED),    # Slight edge
        (0, 1): StrategyWeights(0.55, 0.45, StrategyType.BALANCED),    # Slight edge
        (1, 1): StrategyWeights(0.50, 0.50, StrategyType.BALANCED),    # True neutral
        (2, 1): StrategyWeights(0.45, 0.55, StrategyType.BALANCED),    # Getting careful

        # Hitter's counts - Conservative (avoid walks/damage)
        (2, 0): StrategyWeights(0.40, 0.60, StrategyType.CONSERVATIVE),  # Need strike
        (3, 0): StrategyWeights(0.20, 0.80, StrategyType.CONSERVATIVE),  # Must throw strike
        (3, 1): StrategyWeights(0.25, 0.75, StrategyType.CONSERVATIVE),  # Very careful
        (3, 2): StrategyWeights(0.50, 0.50, StrategyType.BALANCED),      # Full count - balanced
    }

    @classmethod
    def get_weights(cls, balls: int, strikes: int) -> StrategyWeights:
        """
        Get strategy weights for a specific count

        Args:
            balls: Number of balls (0-3)
            strikes: Number of strikes (0-2)

        Returns:
            StrategyWeights with appropriate weights and strategy type

        Raises:
            ValueError: If count is invalid
        """
        if not (0 <= balls <= 3 and 0 <= strikes <= 2):
            raise ValueError(f"Invalid count: {balls}-{strikes}")

        count_key = (balls, strikes)
        if count_key not in cls.MATRIX:
            raise ValueError(f"Count {balls}-{strikes} not in strategy matrix")

        return cls.MATRIX[count_key]

    @classmethod
    def get_all_counts(cls) -> List[Tuple[int, int]]:
        """Get all valid counts in the matrix"""
        return list(cls.MATRIX.keys())


# ============================================================================
# AUTO-TARGETING STRATEGY - HEATMAP
# ============================================================================

# Sweeper Targeting Constants (Data-Driven, No Magic Numbers)
# Strike zone edge: approximately 10 inches (0.83 feet) from center
ZONE_EDGE_X: float = 0.83

# Aggressive offset: aim 6 inches (0.5 feet) outside zone for chase
SWEEPER_OFFSET_AGGRESSIVE: float = 0.5

# Conservative offset: aim 2 inches (0.2 feet) inside edge for backdoor/strike
SWEEPER_OFFSET_CONSERVATIVE: float = -0.2


def calculate_dynamic_target(
    pitch_spec: Dict[str, float],
    strategy: StrategyType,
    pitcher_hand: str = 'R'
) -> Tuple[float, float]:
    """
    Calculate dynamic target location for Sweepers (ST) based on expected pitch movement

    NO HARDCODED COORDINATES - Uses physics-based calculation from spin data.

    Algorithm:
        1. Estimate horizontal break from spin_direction and spin_efficiency
        2. Calculate break_sign based on pitcher handedness and spin
        3. Apply strategy-based offset relative to zone edge

    Args:
        pitch_spec: Dict with spin_direction, spin_efficiency, velocity, etc.
        strategy: Current count strategy (Aggressive, Conservative, Balanced)
        pitcher_hand: Pitcher handedness ('L' or 'R')

    Returns:
        (x, z) target coordinates calculated from physics

    Physics Logic:
        - Sweepers have massive horizontal break (10-20 inches)
        - Break direction depends on spin_direction angle
        - Must aim *against* the break to let it sweep into/out of zone

    Examples:
        >>> # RHP Sweeper with 45° spin_direction (glove-side break)
        >>> calculate_dynamic_target({'spin_direction': 45, 'spin_efficiency': 0.95},
        ...                          StrategyType.AGGRESSIVE, 'R')
        (-1.33, 2.0)  # Aims inside, breaks away for chase
    """
    # Extract pitch physics
    spin_direction: float = pitch_spec.get('spin_direction', 180.0)  # degrees
    spin_efficiency: float = pitch_spec.get('spin_efficiency', 0.95)

    # Step 1: Calculate expected break direction
    # Spin direction 0-180° = glove-side break (RHP: left, LHP: right)
    # Spin direction 180-360° = arm-side break (RHP: right, LHP: left)

    # For RHP: spin_direction < 180 = breaks left (negative x)
    # For LHP: spin_direction < 180 = breaks right (positive x)
    if pitcher_hand == 'R':
        # RHP: Low spin_direction (0-90°) = glove-side (left) = negative break
        break_sign: int = -1 if spin_direction < 180 else 1
    else:
        # LHP: Low spin_direction (0-90°) = glove-side (right) = positive break
        break_sign: int = 1 if spin_direction < 180 else -1

    # Step 2: Estimate break magnitude (in feet)
    # Typical sweeper: 12-18 inches horizontal movement = 1.0-1.5 feet
    # Scale by spin_efficiency (higher efficiency = more break)
    base_break: float = 1.3  # feet (average sweeper break)
    estimated_break: float = base_break * spin_efficiency

    # Step 3: Apply Break-Adjusted Targeting Algorithm
    if strategy == StrategyType.AGGRESSIVE:
        # CHASE STRATEGY: Start in zone, break out for chase
        # Target_X = Break_Direction_Sign * (ZONE_EDGE_X + SWEEPER_OFFSET_AGGRESSIVE)
        # Goal: Aim 6 inches outside zone edge to induce chase
        target_x: float = break_sign * (ZONE_EDGE_X + SWEEPER_OFFSET_AGGRESSIVE)
        target_z: float = 2.0  # Low (sweepers typically thrown low)

    elif strategy == StrategyType.CONSERVATIVE:
        # BACKDOOR/FRONTDOOR STRATEGY: Start outside, break in for strike
        # Target_X = Break_Direction_Sign * (ZONE_EDGE_X + SWEEPER_OFFSET_CONSERVATIVE)
        # Goal: Aim 2 inches inside edge to catch corner or backdoor
        target_x: float = break_sign * (ZONE_EDGE_X + SWEEPER_OFFSET_CONSERVATIVE)
        target_z: float = 2.3  # Slightly higher for better strike probability

    else:  # BALANCED
        # MIXED APPROACH: Aim at zone edge, let it break slightly out
        # Target = zone edge (pitch starts at edge, breaks slightly off)
        target_x: float = break_sign * ZONE_EDGE_X
        target_z: float = 2.1  # Middle-low

    return (target_x, target_z)


class TargetHeatmap:
    """
    Optimal pitch location targets based on pitch type and game strategy

    Coordinates:
        x: Horizontal position (feet from plate center, catcher's view)
           - Negative = inside to RHH / outside to LHH
           - Positive = outside to RHH / inside to LHH
        z: Vertical position (feet from ground)
           - MLB strike zone: ~1.6 to 3.4 feet

    Strategy-Based Targeting:
        - Aggressive: Attack zone edges, chase pitches, high velocity locations
        - Conservative: Paint corners, stay in zone, minimize damage
        - Balanced: Mix of attack and control
    """

    # Target coordinates: (pitch_type, strategy, handedness_matchup) -> (x, z)
    TARGETS: Dict[Tuple[str, StrategyType, str], Tuple[float, float]] = {
        # FASTBALL (FF, FT, SI) - High Heat / Corners
        ('FF', StrategyType.AGGRESSIVE, 'same'): (0.0, 3.3),      # High heat (RHP vs RHH, LHP vs LHH)
        ('FF', StrategyType.AGGRESSIVE, 'opp'): (0.5, 3.2),       # High-away (RHP vs LHH, LHP vs RHH)
        ('FF', StrategyType.CONSERVATIVE, 'same'): (0.3, 2.5),    # Away-middle
        ('FF', StrategyType.CONSERVATIVE, 'opp'): (-0.3, 2.5),    # Inside-middle
        ('FF', StrategyType.BALANCED, 'any'): (0.0, 2.8),         # Middle-up

        ('FT', StrategyType.AGGRESSIVE, 'same'): (-0.4, 2.8),     # Inside corner (two-seam)
        ('FT', StrategyType.AGGRESSIVE, 'opp'): (0.6, 2.8),       # Away (runs away from hitter)
        ('FT', StrategyType.CONSERVATIVE, 'any'): (0.0, 2.5),     # Middle
        ('FT', StrategyType.BALANCED, 'any'): (0.2, 2.6),         # Slight away

        ('SI', StrategyType.AGGRESSIVE, 'same'): (-0.5, 2.2),     # Inside-low (induces weak contact)
        ('SI', StrategyType.AGGRESSIVE, 'opp'): (0.5, 2.2),       # Away-low
        ('SI', StrategyType.CONSERVATIVE, 'any'): (0.0, 2.3),     # Middle-low
        ('SI', StrategyType.BALANCED, 'any'): (0.2, 2.4),         # Slight away-low

        # BREAKING BALLS - Chase or Corner
        ('SL', StrategyType.AGGRESSIVE, 'same'): (0.8, 1.8),      # Low-away chase (RHP vs RHH)
        ('SL', StrategyType.AGGRESSIVE, 'opp'): (-0.6, 2.0),      # Backdoor (RHP vs LHH)
        ('SL', StrategyType.CONSERVATIVE, 'same'): (0.4, 2.2),    # Away-corner
        ('SL', StrategyType.CONSERVATIVE, 'opp'): (-0.3, 2.3),    # Inside-corner
        ('SL', StrategyType.BALANCED, 'same'): (0.5, 2.0),        # Away-low
        ('SL', StrategyType.BALANCED, 'opp'): (-0.4, 2.1),        # Inside-low

        ('CU', StrategyType.AGGRESSIVE, 'any'): (0.2, 1.5),       # Bury it (chase below zone)
        ('CU', StrategyType.CONSERVATIVE, 'any'): (0.0, 2.0),     # Low-zone
        ('CU', StrategyType.BALANCED, 'any'): (0.1, 1.8),         # Low-middle

        ('CB', StrategyType.AGGRESSIVE, 'any'): (0.3, 1.4),       # Low-away chase
        ('CB', StrategyType.CONSERVATIVE, 'any'): (0.0, 1.9),     # Bottom zone
        ('CB', StrategyType.BALANCED, 'any'): (0.2, 1.7),         # Low

        # CHANGEUP - Deception
        ('CH', StrategyType.AGGRESSIVE, 'same'): (0.6, 1.9),      # Away-low (same as FB tunnel)
        ('CH', StrategyType.AGGRESSIVE, 'opp'): (-0.5, 2.0),      # Inside-low
        ('CH', StrategyType.CONSERVATIVE, 'any'): (0.0, 2.2),     # Middle-low
        ('CH', StrategyType.BALANCED, 'any'): (0.3, 2.1),         # Away-low

        # SPLITTER
        ('FS', StrategyType.AGGRESSIVE, 'any'): (0.0, 1.6),       # Bottom zone (drop)
        ('FS', StrategyType.CONSERVATIVE, 'any'): (0.0, 2.1),     # Low-middle
        ('FS', StrategyType.BALANCED, 'any'): (0.0, 1.9),         # Low

        # DEFAULT FALLBACK
        ('DEFAULT', StrategyType.AGGRESSIVE, 'any'): (0.0, 3.0),  # Middle-up
        ('DEFAULT', StrategyType.CONSERVATIVE, 'any'): (0.0, 2.5), # Center
        ('DEFAULT', StrategyType.BALANCED, 'any'): (0.0, 2.6),    # Middle-up
    }

    @classmethod
    def get_optimal_target(
        cls,
        pitch_type: str,
        strategy: StrategyType,
        pitcher_hand: str = 'R',
        batter_hand: str = 'R',
        pitch_spec: Optional[Dict[str, float]] = None
    ) -> Tuple[float, float]:
        """
        Get optimal target location for a pitch based on strategy and matchup

        For Sweepers (ST), uses dynamic physics-based calculation.
        For other pitches, uses static heatmap lookup.

        Args:
            pitch_type: Pitch abbreviation (FF, SL, CU, CH, ST, etc.)
            strategy: Current count strategy (Aggressive, Conservative, Balanced)
            pitcher_hand: Pitcher handedness ('L' or 'R')
            batter_hand: Batter handedness ('L' or 'R')
            pitch_spec: Pitch specification dict (required for ST dynamic targeting)

        Returns:
            (x, z) target coordinates in feet

        Examples:
            >>> get_optimal_target('FF', StrategyType.AGGRESSIVE, 'R', 'R')
            (0.0, 3.3)  # High heat for RHP vs RHH

            >>> get_optimal_target('ST', StrategyType.AGGRESSIVE, 'R', 'R',
            ...                    {'spin_direction': 45, 'spin_efficiency': 0.95})
            (-1.33, 2.0)  # Dynamic calculation for sweeper (chase strategy)
        """
        # Normalize pitch type
        pitch_type_norm = pitch_type.upper().strip()

        # DYNAMIC TARGETING: Sweepers (ST) use physics-based calculation
        if pitch_type_norm == 'ST':
            if pitch_spec is None:
                # Fallback if no pitch_spec provided
                pitch_spec = {'spin_direction': 45, 'spin_efficiency': 0.95}
            return calculate_dynamic_target(pitch_spec, strategy, pitcher_hand)

        # STATIC TARGETING: All other pitches use heatmap lookup
        # Determine handedness matchup
        if pitcher_hand == batter_hand:
            matchup = 'same'  # Same-handed (harder for batter)
        else:
            matchup = 'opp'   # Opposite-handed

        # Try exact match first
        key = (pitch_type_norm, strategy, matchup)
        if key in cls.TARGETS:
            return cls.TARGETS[key]

        # Try matchup-agnostic ('any')
        key_any = (pitch_type_norm, strategy, 'any')
        if key_any in cls.TARGETS:
            return cls.TARGETS[key_any]

        # Try default for this strategy
        default_key = ('DEFAULT', strategy, 'any')
        if default_key in cls.TARGETS:
            return cls.TARGETS[default_key]

        # Absolute fallback: center of strike zone
        return (0.0, 2.5)


@dataclass
@dataclass
class RankedPitch:
    """
    Ranked pitch recommendation with scoring details

    Attributes:
        pitch_id: Unique pitch identifier
        pitch_type: Human-readable pitch name
        rank: Recommendation rank (1 = best)
        score: Calculated weighted score (lower is better)
        strategy_used: Strategy type applied
        rationale: Natural language explanation
        usage_percentage: GTO-based recommended usage % (0.0-100.0)

        # Raw metrics for transparency
        xrv: Expected run value (negative = good)
        var_95: Value at Risk (95th percentile)
        strike_probability: Probability of strike
        context_adjusted_xrv: Leverage-adjusted xRV
    """
    pitch_id: str
    pitch_type: str
    rank: int
    score: float
    strategy_used: StrategyType
    rationale: str
    usage_percentage: float  # NEW: Softmax-based usage recommendation

    # Raw metrics (for transparency)
    xrv: float
    var_95: float
    strike_probability: float
    context_adjusted_xrv: float

    def to_dict(self) -> dict:
        """Convert to dictionary for API response"""
        return {
            'pitch_id': self.pitch_id,
            'pitch_type': self.pitch_type,
            'rank': self.rank,
            'score': self.score,
            'strategy_used': self.strategy_used.value,
            'rationale': self.rationale,
            'usage_percentage': self.usage_percentage,  # NEW
            'metrics': {
                'xrv': self.xrv,
                'var_95': self.var_95,
                'strike_probability': self.strike_probability,
                'context_adjusted_xrv': self.context_adjusted_xrv
            }
        }


def calculate_pitch_score(
    xrv: float,
    var_95: float,
    balls: int,
    strikes: int
) -> Tuple[float, StrategyType]:
    """
    Calculate pitch score using count-specific strategy

    Formula:
        Score = w_stuff × xRV + w_control × VaR

    Where:
        - w_stuff: Weight for performance/stuff (from StrategyMatrix)
        - w_control: Weight for command/risk (from StrategyMatrix)
        - Lower score is BETTER (negative xRV is good for pitcher)

    Args:
        xrv: Expected run value (mean_xrv from simulation)
        var_95: Value at Risk - 95th percentile worst outcome
        balls: Ball count (0-3)
        strikes: Strike count (0-2)

    Returns:
        (score, strategy_type): Weighted score and strategy used

    Example:
        >>> # 0-2 count (aggressive): prioritize strikeout potential
        >>> score, strategy = calculate_pitch_score(-0.75, -0.45, 0, 2)
        >>> print(f"Score: {score:.3f}, Strategy: {strategy}")
        Score: -0.690, Strategy: Aggressive

        >>> # 3-0 count (conservative): prioritize strikes
        >>> score, strategy = calculate_pitch_score(-0.75, -0.45, 3, 0)
        >>> print(f"Score: {score:.3f}, Strategy: {strategy}")
        Score: -0.510, Strategy: Conservative
    """
    # Get count-specific weights
    weights = StrategyMatrix.get_weights(balls, strikes)

    # Calculate weighted score
    # Note: Both xRV and VaR are negative for good outcomes
    score = (weights.w_stuff * xrv) + (weights.w_control * var_95)

    return score, weights.strategy


def generate_rationale(
    pitch_type: str,
    balls: int,
    strikes: int,
    strike_probability: float,
    xrv: float,
    var_95: float,
    strategy: StrategyType,
    rank: int
) -> str:
    """
    Generate natural language rationale for pitch recommendation

    Considers:
        - Count situation
        - Strike probability
        - Pitch performance (xRV)
        - Risk profile (VaR)
        - Overall strategy

    Args:
        pitch_type: Name of pitch
        balls: Ball count
        strikes: Strike count
        strike_probability: Probability of strike (0-1)
        xrv: Expected run value
        var_95: Value at Risk
        strategy: Strategy type used
        rank: Recommendation rank

    Returns:
        Natural language explanation string
    """
    count_str = f"{balls}-{strikes}"

    # Determine pitch characteristics
    is_high_strike_rate = strike_probability > 0.65
    is_elite_stuff = xrv < -0.70  # Very negative xRV is elite
    is_risky = var_95 > -0.50      # Less negative VaR is risky

    # Build rationale based on strategy and pitch characteristics
    rationale_parts = []

    # Rank context
    if rank == 1:
        rationale_parts.append("**Primary recommendation:**")
    elif rank == 2:
        rationale_parts.append("**Secondary option:**")
    else:
        rationale_parts.append(f"**Alternative #{rank}:**")

    # Count-specific logic
    if strategy == StrategyType.AGGRESSIVE:
        # Pitcher's count - prioritize whiff
        if is_elite_stuff:
            rationale_parts.append(f"Elite strikeout pitch for {count_str} count.")
            if strikes == 2:
                rationale_parts.append("Two-strike situation - go for the punchout.")
        else:
            rationale_parts.append(f"Solid chase option on {count_str}.")
            rationale_parts.append("Can expand zone to induce weak contact.")

    elif strategy == StrategyType.CONSERVATIVE:
        # Hitter's count - prioritize strikes
        if is_high_strike_rate:
            rationale_parts.append(f"High strike probability ({strike_probability:.0%}) critical on {count_str}.")
            if balls == 3:
                rationale_parts.append("Safe bet to avoid walk.")
        else:
            rationale_parts.append(f"Moderate strike rate on {count_str} - use with caution.")
            if balls == 3:
                rationale_parts.append("Consider as fallback to avoid walk.")

        if is_risky:
            rationale_parts.append("⚠️ Higher variance - command critical.")

    else:  # BALANCED
        # Neutral count - balanced approach
        if is_elite_stuff and is_high_strike_rate:
            rationale_parts.append(f"Best of both worlds - elite stuff + {strike_probability:.0%} strikes.")
        elif is_elite_stuff:
            rationale_parts.append(f"Strong whiff potential, but locate carefully.")
        elif is_high_strike_rate:
            rationale_parts.append(f"Reliable strike-getter ({strike_probability:.0%}).")
        else:
            rationale_parts.append(f"Balanced option for {count_str} count.")

    # Performance summary
    if xrv < -0.70:
        performance_desc = "Elite performance"
    elif xrv < -0.60:
        performance_desc = "Above-average performance"
    else:
        performance_desc = "Average performance"

    rationale_parts.append(f"{performance_desc} (xRV: {xrv:.3f}).")

    return " ".join(rationale_parts)


def calculate_usage_percentages(
    scored_pitches: List[dict],
    temperature: float = None
) -> List[float]:
    """
    Calculate GTO-based usage percentages using numerically stable Softmax.

    Implements the Log-Sum-Exp trick for numerical stability to prevent
    underflow/overflow in exponential calculations.

    Philosophy:
        - Lower scores are BETTER (negative xRV is good for pitcher)
        - Softmax converts scores to probability distribution
        - Temperature controls sharpness (see SOFTMAX_TEMPERATURE constant)

    Algorithm:
        1. Filter out waste pitches (strike_rate < 10%)
        2. Invert scores: logit = -score (lower score → higher logit)
        3. Apply temperature scaling: z = logit / T
        4. Numerically stable softmax: P_i = exp(z_i - z_max) / sum(exp(z_j - z_max))
        5. Normalize to percentages and ensure sum = 100.0

    Args:
        scored_pitches: List of dicts with 'score' and 'strike_prob' keys
        temperature: Softmax temperature (defaults to SOFTMAX_TEMPERATURE)

    Returns:
        List of usage percentages (0.0-100.0), same order as input
        Sum is guaranteed to equal 100.0 (floating point remainder assigned to top pitch)

    Edge Cases:
        - Waste pitches (strike_rate < 10%): Forced to 0.0%
        - All pitches invalid: Return zeros
        - Single valid pitch: Gets 100.0%
    """
    if not scored_pitches:
        return []

    # Use module-level constant if temperature not specified
    if temperature is None:
        temperature = SOFTMAX_TEMPERATURE

    n_pitches = len(scored_pitches)

    # ========================================================================
    # STEP 1: Filter out non-competitive pitches
    # ========================================================================
    # Waste pitches (strike rate < 10%) should not be recommended
    valid_indices = [
        i for i, p in enumerate(scored_pitches)
        if p.get('strike_prob', 0.0) >= MIN_STRIKE_RATE_THRESHOLD
    ]

    # Edge case: All pitches are waste pitches
    if not valid_indices:
        return [0.0] * n_pitches

    # Edge case: Only one valid pitch
    if len(valid_indices) == 1:
        percentages = [0.0] * n_pitches
        percentages[valid_indices[0]] = 100.0
        return percentages

    # ========================================================================
    # STEP 2-3: Invert scores and apply temperature scaling
    # ========================================================================
    # Lower score is BETTER → invert sign so higher logit = better pitch
    logits = []
    for i in valid_indices:
        score = scored_pitches[i].get('score', 0.0)
        logit = -1.0 * score  # Invert (critical for correct ranking)
        weighted_logit = logit / temperature  # Temperature scaling
        logits.append(weighted_logit)

    # ========================================================================
    # STEP 4: Numerically stable softmax (Log-Sum-Exp trick)
    # ========================================================================
    # Subtract max to prevent overflow: exp(x - max) instead of exp(x)
    max_logit = max(logits)

    # Compute exponentials (now numerically stable)
    exp_vals = [math.exp(logit - max_logit) for logit in logits]

    # Sum of exponentials
    sum_exp = sum(exp_vals)

    # Edge case: Sum is zero (should never happen with proper data, but handle gracefully)
    if sum_exp == 0.0 or math.isinf(sum_exp) or math.isnan(sum_exp):
        # Fallback to uniform distribution among valid pitches
        uniform_pct = 100.0 / len(valid_indices)
        percentages = [0.0] * n_pitches
        for idx in valid_indices:
            percentages[idx] = uniform_pct
        return percentages

    # Compute percentages
    percentages_valid = [(exp_val / sum_exp) * 100.0 for exp_val in exp_vals]

    # ========================================================================
    # STEP 5: Map back to original indices and ensure exact sum = 100.0
    # ========================================================================
    percentages = [0.0] * n_pitches
    for idx, valid_idx in enumerate(valid_indices):
        percentages[valid_idx] = percentages_valid[idx]

    # Handle floating point rounding errors
    # Assign remainder to the top-ranked pitch (first valid index after sorting)
    actual_sum = sum(percentages)
    if actual_sum > 0:
        remainder = 100.0 - actual_sum
        # Find the pitch with highest percentage (should be first valid index)
        top_pitch_idx = valid_indices[0]
        percentages[top_pitch_idx] += remainder

    return percentages


class PitchRecommendationService:
    """
    High-level service for generating pitch recommendations

    Orchestrates the Commander Logic to rank pitches based on:
    - Count situation
    - Pitch performance (xRV)
    - Command/risk (VaR)
    - Game context

    Usage:
        >>> service = PitchRecommendationService()
        >>> analyses = [...] # From BatchSimulationService
        >>> ranked = service.rank_pitches(analyses, balls=0, strikes=2)
        >>> for pitch in ranked:
        >>>     print(f"{pitch.rank}. {pitch.pitch_type}: {pitch.rationale}")
    """

    def rank_pitches(
        self,
        pitch_analyses: List,
        balls: int,
        strikes: int
    ) -> List[RankedPitch]:
        """
        Rank pitches using Commander Logic with GTO usage percentages

        Args:
            pitch_analyses: List of PitchAnalysis objects from batch simulation
            balls: Ball count (0-3)
            strikes: Strike count (0-2)

        Returns:
            List of RankedPitch objects sorted by score (best first)
            with usage_percentage calculated via softmax

        Note:
            Lower score is better (negative xRV/VaR is good for pitcher)
        """
        # Calculate score for each pitch
        scored_pitches = []

        for analysis in pitch_analyses:
            # Extract metrics
            xrv = analysis.risk_metrics.mean_xrv
            var_95 = analysis.risk_metrics.var_95
            strike_prob = analysis.strike_probability
            context_xrv = analysis.context_adjusted_xrv

            # Calculate score using Commander Logic
            score, strategy = calculate_pitch_score(xrv, var_95, balls, strikes)

            scored_pitches.append({
                'analysis': analysis,
                'score': score,
                'strategy': strategy,
                'xrv': xrv,
                'var_95': var_95,
                'strike_prob': strike_prob,
                'context_xrv': context_xrv
            })

        # Sort by score (lower is better)
        scored_pitches.sort(key=lambda x: x['score'])

        # ========================================================================
        # GTO-BASED USAGE PERCENTAGES (Softmax with Temperature)
        # ========================================================================
        usage_percentages = calculate_usage_percentages(scored_pitches, temperature=0.5)

        # Generate ranked recommendations
        ranked = []
        for rank, (item, usage_pct) in enumerate(zip(scored_pitches, usage_percentages), start=1):
            analysis = item['analysis']

            # Generate rationale
            rationale = generate_rationale(
                pitch_type=analysis.pitch_type,
                balls=balls,
                strikes=strikes,
                strike_probability=item['strike_prob'],
                xrv=item['xrv'],
                var_95=item['var_95'],
                strategy=item['strategy'],
                rank=rank
            )

            # Create RankedPitch object with usage percentage
            ranked_pitch = RankedPitch(
                pitch_id=analysis.pitch_id,
                pitch_type=analysis.pitch_type,
                rank=rank,
                score=item['score'],
                strategy_used=item['strategy'],
                rationale=rationale,
                usage_percentage=usage_pct,  # NEW: GTO-based usage
                xrv=item['xrv'],
                var_95=item['var_95'],
                strike_probability=item['strike_prob'],
                context_adjusted_xrv=item['context_xrv']
            )

            ranked.append(ranked_pitch)

        return ranked

    def get_primary_recommendation(
        self,
        pitch_analyses: List,
        balls: int,
        strikes: int
    ) -> Optional[RankedPitch]:
        """
        Get the top-ranked pitch recommendation

        Args:
            pitch_analyses: List of PitchAnalysis objects
            balls: Ball count
            strikes: Strike count

        Returns:
            Top-ranked RankedPitch or None if no analyses provided
        """
        ranked = self.rank_pitches(pitch_analyses, balls, strikes)
        return ranked[0] if ranked else None

    def get_strategy_summary(self, balls: int, strikes: int) -> dict:
        """
        Get strategy summary for a specific count

        Args:
            balls: Ball count (0-3)
            strikes: Strike count (0-2)

        Returns:
            Dictionary with strategy details
        """
        weights = StrategyMatrix.get_weights(balls, strikes)

        return {
            'count': f"{balls}-{strikes}",
            'strategy': weights.strategy.value,
            'w_stuff': weights.w_stuff,
            'w_control': weights.w_control,
            'description': self._get_strategy_description(weights.strategy)
        }

    @staticmethod
    def _get_strategy_description(strategy: StrategyType) -> str:
        """Get human-readable strategy description"""
        descriptions = {
            StrategyType.AGGRESSIVE: "Attack mode - prioritize strikeout potential",
            StrategyType.BALANCED: "Neutral approach - balance stuff and control",
            StrategyType.CONSERVATIVE: "Defensive mode - prioritize strikes, avoid walks"
        }
        return descriptions[strategy]
