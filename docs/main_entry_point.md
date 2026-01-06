# Aegis Pitching Engine - Main Entry Point

## Overview

`src/main.py` is the **entry point** of the Aegis Pitching Engine project. It integrates all modules to perform a **single at-bat simulation** featuring a high-leverage scenario between Walker Buehler and Shohei Ohtani.

---

## Features

### 1. **Complete Module Integration**

- ✅ **AegisDataLoader**: Loads real pitcher data from DuckDB
- ✅ **AegisStrategyEngine**: Makes pitch recommendations
- ✅ **ContextEncoder**: Encodes game state
- ✅ **EntropyMonitor**: Tracks pitch pattern entropy
- ✅ **TunnelingAnalyzer**: Evaluates pitch sequencing
- ✅ **EffectiveVelocityCalculator**: Computes perceived velocity

### 2. **Realistic Scenario**

- **Pitcher**: Walker Buehler (ID: 621111)
- **Batter**: Shohei Ohtani (ID: 660271)
- **Situation**: Bottom 9th, 2 outs, bases loaded, 3-2 count
- **Leverage**: 🔴 CRITICAL - High Leverage Situation
- **Pitch Count**: 98 (Fatigue Critical)

### 3. **Data-Driven Decision**

- Loads Walker Buehler's **2024 pitch usage** from Baseball Savant data
- Filters ghost pitches (< 3% usage)
- Applies sample size penalties for unreliable statistics
- Generates natural language rationale with usage rates

### 4. **Comprehensive Output**

- **Situation Report**: Game context and matchup details
- **AI Recommendation**: Pitch type, zone, location, probabilities
- **Strategic Rationale**: Natural language explanation
- **Physics Visualization**: Trajectory plot (placeholder)
- **Detailed Logging**: `aegis_simulation.log`

---

## Usage

### Basic Execution

```bash
cd /Users/ekim56/Desktop/aegis-pitching-engine
python src/main.py
```

### Expected Output

```
================================================================================
📋 SITUATION REPORT - The War Room
================================================================================

🏟️  Scenario: Walker Buehler vs. Shohei Ohtani
   Inning: Bottom 9th
   Outs: 2
   Count: 3-2
   Runners: Bases Loaded (1st, 2nd, 3rd)
   Score: Leading by 1 run(s)
   Leverage: 🔴 CRITICAL - High Leverage Situation

⚾ Pitcher Status:
   Hand: R
   Pitch Count: 98 (Fatigue Critical)
   Entropy: 0.62
   Previous Pitch: FF @ 97.0 mph

🎯 Batter Profile:
   Hand: L
   Chase Rate: 32.0%
   Whiff Rate: 28.0%
   ISO: 0.350 (⚠️  HIGH POWER)
   OPS: 1.050
   GB/FB: 0.80

================================================================================
🤖 AI RECOMMENDATION
================================================================================

✅ Recommended Pitch:
   Type: SI
   Zone: shadow_out_low
   Location: (0.58, 2.00)

📊 Top 3 Action Probabilities:
   1. SI_chase_out: 1.8%
   2. CH_chase_out: 1.7%
   3. CH_chase_low: 1.7%

================================================================================
📝 STRATEGIC RATIONALE
================================================================================

변화구 Sinker(15%)로, 직전 Four-Seam Fastball(FF) 이후, EV 차이가 +7.9mph로 크며,
Sinker(SI)를 shadow_out_low 존에 선택함, (주의: 데이터 신뢰도 50%),
현재 승부처 상황으로 확실한 공을 선택했습니다.

💾 Physics Visualization: simulation_result.png
```

---

## Architecture

### Execution Flow

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Setup & Config                                           │
│    - Load StrategyConfig                                    │
│    - Setup logging (INFO level)                             │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. Scenario Definition                                      │
│    - Game state (9th inning, 2 outs, bases loaded)         │
│    - Pitcher state (98 pitch count, fatigue)               │
│    - Matchup state (Ohtani's statistics)                   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. Data Loading                                             │
│    - AegisDataLoader connects to DuckDB                     │
│    - Load Buehler's 2024 pitch usage (15,419 pitches)      │
│    - Calculate stuff_plus, sample_sizes, zone_command      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. Engine Execution                                         │
│    - Initialize AegisStrategyEngine                         │
│    - Filter ghost pitches (e.g., SL with 1% usage)         │
│    - Calculate metrics (tunneling, EV, stuff+, entropy)    │
│    - Apply Softmax with high-leverage temperature           │
│    - Generate rationale with usage rates                    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. Results Display                                          │
│    - Print AI recommendation                                │
│    - Print strategic rationale                              │
│    - Save visualization (simulation_result.png)             │
└─────────────────────────────────────────────────────────────┘
```

---

## Real Data Integration

### Walker Buehler (2024 Stats)

When DuckDB is available, `main.py` loads **real Baseball Savant data**:

```python
# Actual pitch usage from 15,419 pitches
pitch_usage_stats = {
    'FF': 0.30,   # Four-Seam Fastball (30.1%)
    'FC': 0.20,   # Cutter (20.1%)
    'KC': 0.19,   # Knuckle Curve (19.4%)
    'SI': 0.15,   # Sinker (14.9%)
    'ST': 0.10,   # Sweeper (9.6%)
    'CH': 0.05,   # Changeup (4.6%)
    'SL': 0.01    # Slider (1.0%) <- Filtered as noise!
}
```

**Note**: SL (Slider at 1.0%) is automatically filtered by the noise robustness system.

### Fallback Behavior

If DuckDB is unavailable, the system uses default stats:

```python
pitch_usage_stats = {
    'FF': 0.55,
    'SL': 0.28,
    'CU': 0.10,
    'CH': 0.07
}
```

---

## Output Files

### 1. `simulation_result.png`

Trajectory visualization placeholder (46 KB PNG)

To implement full visualization:

```python
# Future enhancement
from src.game_theory.tunneling import TunnelingAnalyzer

analyzer = TunnelingAnalyzer()
trajectory = analyzer.simulate_trajectory(action, pitcher_state)
analyzer.plot_trajectory(trajectory, output_path='simulation_result.png')
```

### 2. `aegis_simulation.log`

Detailed execution log with timestamps (INFO level)

Example entries:

```
2026-01-06 11:13:37,474 - __main__ - INFO - Step 1: Setup & Configuration
2026-01-06 11:13:37,498 - __main__ - INFO - 📊 투수 데이터 로딩: ID=621111
2026-01-06 11:13:37,633 - src.game_theory.engine - WARNING - Ignored noise pitch: SL
2026-01-06 11:13:37,635 - __main__ - INFO - ✅ 의사결정 완료
```

---

## Customization

### Change the Scenario

Edit the scenario in `main()` function:

```python
# Game state
game_state = {
    'outs': 0,              # Change to 0 outs
    'count': '0-0',         # Change to 0-0 count
    'runners': [0, 0, 0],   # Empty bases
    'score_diff': 5,        # 5 run lead (low leverage)
    'inning': 3             # 3rd inning
}
```

### Use Different Pitcher

```python
pitcher_id = 543037  # Justin Verlander
pitcher_stats = load_pitcher_stats(loader, pitcher_id, year=2024)
```

### Modify Batter Profile

```python
matchup_state = {
    'batter_hand': 'R',       # Right-handed
    'chase_rate': 0.45,       # High chase rate (worse discipline)
    'whiff_rate': 0.35,       # High whiff rate (weaker contact)
    'iso': 0.150,             # Low power
    'gb_fb_ratio': 1.5,       # Ground ball hitter
    'ops': 0.650              # Below average hitter
}
```

---

## Dependencies

### Required Modules

- `src.common.config`: StrategyConfig
- `src.data_pipeline.data_loader`: AegisDataLoader
- `src.game_theory.engine`: AegisStrategyEngine

### Optional Dependencies

- `matplotlib`: For trajectory visualization
- `DuckDB`: For real data loading (works without it)

---

## Error Handling

### No DuckDB File

```
⚠️  DuckDB 파일을 찾을 수 없습니다. 기본값 사용.
```

**Action**: System uses default pitcher stats

### Schema Validation Failure

```
⚠️  스키마 검증 실패. 기본값 사용.
```

**Action**: Falls back to default stats

### Critical Error

```
❌ SIMULATION FAILED
Error: [error details]
See aegis_simulation.log for details.
```

**Action**: Check log file for traceback

---

## Testing

### Verify All Modules Work

```bash
# Test data loader
python src/data_pipeline/data_loader.py

# Test strategy engine
python src/game_theory/engine.py

# Test full integration
python src/main.py
```

### Expected Test Results

1. ✅ DuckDB connection successful
2. ✅ Schema validation passed (24 columns)
3. ✅ 15,419 pitches loaded for Buehler
4. ✅ Ghost pitch filtered (SL at 1.0%)
5. ✅ AI recommendation generated
6. ✅ Rationale includes usage rates
7. ✅ Visualization saved

---

## Performance

### Execution Time

- **Data Loading**: ~0.1s (15,419 pitches)
- **Engine Initialization**: ~0.5s
- **Decision Making**: ~0.01s
- **Total Runtime**: < 1 second

### Memory Usage

- Peak: ~200 MB (includes PyTorch models)

---

## Future Enhancements

1. **Real Batter Data**: Load from DuckDB instead of hardcoded
2. **Full Game Simulation**: Simulate entire 9 innings
3. **Real-Time Trajectory Plot**: 3D visualization with matplotlib
4. **API Endpoint**: FastAPI service for web integration
5. **Model Fine-Tuning**: Train ContextEncoder on historical data
6. **Monte Carlo Simulation**: Run 1000 simulations for outcome distribution

---

## Troubleshooting

### Issue: "FileNotFoundError: DuckDB file not found"

**Solution**: Check path in `src/common/config.py`:

```python
DB_PATH = Path(__file__).parent.parent.parent / "data/01_raw/savant.duckdb"
```

### Issue: "ImportError: No module named 'matplotlib'"

**Solution**: Install matplotlib or run without visualization:

```bash
poetry add matplotlib
# or
pip install matplotlib
```

### Issue: "All pitches filtered"

**Solution**: Lower MIN_PITCH_USAGE_THRESHOLD in config:

```python
MIN_PITCH_USAGE_THRESHOLD = 0.01  # 1% instead of 3%
```

---

## Related Documentation

- [Engine Documentation](../docs/data_noise_robustness.md)
- [Architecture Overview](../docs/architecture.md)
- [Development Conventions](../docs/convention.md)
- [Project Roadmap](../docs/roadmap.md)

---

**Version**: 1.0.0
**Last Updated**: 2026-01-06
**Author**: Aegis Pitching Engine Team
