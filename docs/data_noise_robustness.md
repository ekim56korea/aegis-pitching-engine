# Data Noise Robustness in AegisStrategyEngine

## Overview

AegisStrategyEngine is now **robust to data noise** from sources like Trackman misclassification, small sample sizes, and unreliable statistics. This document explains the implemented features and how to use them.

---

## Key Features

### 1. Ghost Pitch Filtering

**Problem**: Trackman sometimes misclassifies pitches, creating "ghost pitches" with < 3% usage that don't actually exist in a pitcher's repertoire.

**Solution**: `_filter_ghost_pitches()` automatically removes pitches below `MIN_PITCH_USAGE_THRESHOLD` (default: 3%).

**Example**:

```python
pitch_usage_stats = {
    'FF': 0.60,  # 60% - Passes filter
    'SL': 0.30,  # 30% - Passes filter
    'CH': 0.09,  # 9% - Passes filter
    'KN': 0.01   # 1% - FILTERED (< 3%)
}

# KN will be removed with a warning log:
# "Ignored noise pitch: KN (1.0% < 3.0% threshold)"
```

**Fallback**: If all pitches are filtered, the engine force-selects the pitch with highest usage rate to prevent crashes.

---

### 2. Sample Size Penalties

**Problem**: Low sample sizes make Stuff+ calculations unreliable (e.g., 5 curveballs vs. 150 fastballs).

**Solution**: `_calculate_stuff_quality_robust()` applies penalties to Stuff+ scores when:

- Sample size < `MIN_SAMPLE_SIZE_THRESHOLD` (default: 10)
- Usage rate < 10%

**Penalty Formula**:

```python
if sample_count < MIN_SAMPLE_SIZE_THRESHOLD:
    stuff_score *= LOW_SAMPLE_PENALTY  # 0.7 (70% of original)

if usage_rate < 0.10:
    confidence = max(0.5, usage_rate / 0.10)
    stuff_score *= confidence
```

**Example**:

```python
pitcher_stats = {
    'stuff_plus': {'CH': 105.0},
    'sample_sizes': {'CH': 7}  # Only 7 pitches
}

# Original: 105.0
# After penalty: 105.0 * 0.7 = 73.5
```

---

### 3. Usage Rate in Rationale

**Problem**: Users need to understand which pitches are primary vs. experimental.

**Solution**: Rationale now includes usage rates and pitch role classification:

**Output Examples**:

- **Primary Pitch (≥40%)**: "주무기인 Four-Seam Fastball(60%)로..."
- **Secondary Pitch (20-40%)**: "보조 구종인 Slider(28%)로..."
- **Off-Speed (< 20%)**: "변화구 Changeup(12%)로..."

**Data Quality Warning**:

```
Rationale: "변화구 Changeup(8%)로... (주의: 데이터 신뢰도 21%)"
```

---

## Configuration Parameters

All parameters are defined in `src/common/config.py`:

```python
# Data Noise Filtering
MIN_PITCH_USAGE_THRESHOLD = 0.03  # 3% threshold
MIN_SAMPLE_SIZE_THRESHOLD = 10     # Minimum samples for reliable Stuff+
LOW_SAMPLE_PENALTY = 0.7           # 70% multiplier for low-sample pitches
NOISE_LOGGING_ENABLED = True       # Enable/disable warning logs
```

---

## API Changes

### Before (Old API)

```python
result = engine.decide_pitch(
    game_state,
    pitcher_state,
    matchup_state,
    available_pitches=['FF', 'SL', 'CH'],  # ❌ Old: List of strings
    pitcher_stats
)
```

### After (New API)

```python
result = engine.decide_pitch(
    game_state,
    pitcher_state,
    matchup_state,
    pitch_usage_stats={'FF': 0.60, 'SL': 0.30, 'CH': 0.10},  # ✅ New: Dict with usage rates
    pitcher_stats
)
```

### Enhanced DecisionResult

```python
@dataclass
class DecisionResult:
    selected_action: Action
    action_probs: Dict[str, float]
    q_values: Dict[str, float]
    rationale: str
    leverage_level: str
    entropy_status: str
    filtered_pitches: Dict[str, float]  # NEW: Pitches that passed filtering
    noise_pitches: List[str]            # NEW: Pitches removed as noise
```

---

## Usage Example

```python
from src.game_theory.engine import AegisStrategyEngine

engine = AegisStrategyEngine(device='cpu')

# Game state
game_state = {
    'outs': 2,
    'count': '3-2',
    'runners': [1, 1, 1],
    'score_diff': -1,
    'inning': 9
}

# Pitcher state
pitcher_state = {
    'hand': 'R',
    'role': 'RP',
    'pitch_count': 28,
    'entropy': 0.65,
    'prev_pitch': 'SL',
    'prev_velo': 85.0
}

# Matchup
matchup_state = {
    'batter_hand': 'L',
    'times_faced': 1,
    'chase_rate': 0.38,
    'whiff_rate': 0.32,
    'iso': 0.220,
    'gb_fb_ratio': 0.9,
    'ops': 0.810
}

# Pitch usage with ghost pitch
pitch_usage_stats = {
    'FF': 0.55,   # Primary
    'SL': 0.30,   # Secondary
    'CH': 0.145,  # Off-speed
    'KN': 0.005   # Ghost pitch (will be filtered)
}

# Pitcher stats with sample sizes
pitcher_stats = {
    'stuff_plus': {
        'FF': 105.0,
        'SL': 115.0,
        'CH': 98.0,
        'KN': 92.0
    },
    'sample_sizes': {
        'FF': 165,
        'SL': 90,
        'CH': 44,
        'KN': 2  # Too few samples
    }
}

# Make decision
result = engine.decide_pitch(
    game_state,
    pitcher_state,
    matchup_state,
    pitch_usage_stats,
    pitcher_stats
)

# Inspect results
print(f"Selected: {result.selected_action.pitch_type} @ {result.selected_action.zone}")
print(f"Filtered Pitches: {list(result.filtered_pitches.keys())}")
print(f"Noise Pitches: {result.noise_pitches}")
print(f"Rationale: {result.rationale}")
```

**Expected Output**:

```
WARNING - Ignored noise pitch: KN (0.5% < 3.0% threshold). Likely Trackman misclassification.
INFO - Filtered pitches: ['FF', 'SL', 'CH'] (removed 1 noise pitches)

Selected: FF @ chase_high
Filtered Pitches: ['FF', 'SL', 'CH']
Noise Pitches: ['KN']
Rationale: 주무기인 Four-Seam Fastball(55%)로, 직전 Slider(SL) 이후, EV 차이가 +12.5mph로 크며...
```

---

## Logging

Enable/disable noise filtering logs in config:

```python
NOISE_LOGGING_ENABLED = True  # Set to False to suppress logs
```

**Log Levels**:

- **WARNING**: Individual pitch removal (e.g., "Ignored noise pitch: KN")
- **INFO**: Summary of filtering process
- **DEBUG**: Sample size penalty application (only if debugging enabled)

---

## Testing

Run the test suite to verify robustness:

```bash
cd /Users/ekim56/Desktop/aegis-pitching-engine
python src/game_theory/engine.py
```

**Test Case 1**: Ghost pitch filtering (KN with 0.5% usage)
**Test Case 2**: Sample size penalty (CH with 8 samples)

---

## Best Practices

1. **Always provide pitch_usage_stats**: Use actual usage rates from Savant data
2. **Include sample_sizes**: Helps engine assess data quality
3. **Monitor filtered_pitches**: Check if legitimate pitches are being removed
4. **Adjust thresholds if needed**: Some pitchers legitimately throw rare pitches (e.g., screwball)

---

## Troubleshooting

### Problem: Legitimate pitch is filtered

**Solution**: Lower `MIN_PITCH_USAGE_THRESHOLD` in config:

```python
MIN_PITCH_USAGE_THRESHOLD = 0.01  # 1% instead of 3%
```

### Problem: Too many warnings

**Solution**: Disable logging:

```python
NOISE_LOGGING_ENABLED = False
```

### Problem: All pitches filtered

**Cause**: All pitches have < 3% usage (data quality issue)
**Behavior**: Engine force-selects highest usage pitch with warning

---

## Technical Details

### Data Quality Score Formula

```python
def _assess_data_quality(action, pitcher_stats, pitch_usage):
    """
    Returns quality score [0, 1] based on:
    1. Usage rate (higher = more reliable)
    2. Sample size (more samples = more reliable)
    """
    usage_rate = pitch_usage.get(action.pitch_type, 0.0)
    sample_count = pitcher_stats['sample_sizes'].get(action.pitch_type, 0)

    # Usage component (weight: 0.6)
    usage_quality = min(1.0, usage_rate / 0.30)

    # Sample size component (weight: 0.4)
    sample_quality = min(1.0, sample_count / 50)

    return 0.6 * usage_quality + 0.4 * sample_quality
```

### Fallback Logic

```python
if not filtered_pitches:
    # Force-select primary pitch to prevent crashes
    primary_pitch = max(pitch_usage_stats, key=pitch_usage_stats.get)
    filtered_pitches = {primary_pitch: pitch_usage_stats[primary_pitch]}
    logger.warning(f"All pitches filtered! Force-selected primary: {primary_pitch}")
```

---

## Future Enhancements

1. **Bayesian Priors**: Use league-average Stuff+ as prior when sample size is small
2. **Confidence Intervals**: Return uncertainty ranges for low-sample predictions
3. **Adaptive Thresholds**: Adjust MIN_PITCH_USAGE_THRESHOLD based on pitcher role (SP vs RP)
4. **Temporal Filtering**: Ignore pitches not thrown in last N games

---

**Version**: 1.0.0
**Last Updated**: 2026-01-06
**Author**: Aegis Pitching Engine Team
