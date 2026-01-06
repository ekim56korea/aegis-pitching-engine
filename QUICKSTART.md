# Aegis Pitching Engine - Quick Start Guide

## 🚀 Complete Project Execution

This guide walks you through running the entire Aegis Pitching Engine from scratch.

---

## Prerequisites

### 1. Python Environment

```bash
# Verify Python version (3.10+)
python --version  # Should show Python 3.10 or higher

# Create virtual environment (if not exists)
python -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate     # Windows
```

### 2. Install Dependencies

```bash
# Install via pip
pip install torch numpy pandas scipy duckdb

# Or via poetry (if using)
poetry install
```

---

## Project Structure

```
aegis-pitching-engine/
├── src/
│   ├── main.py                    # 🎯 ENTRY POINT (START HERE)
│   ├── common/
│   │   └── config.py              # Configuration parameters
│   ├── data_pipeline/
│   │   └── data_loader.py         # DuckDB data loading
│   ├── game_theory/
│   │   ├── engine.py              # Main decision engine
│   │   ├── context_encoder.py    # Game state encoding
│   │   ├── entropy.py             # Pattern monitoring
│   │   ├── effective_velocity.py # Perceived velocity
│   │   └── tunneling.py           # Pitch sequencing
│   ├── physics_engine/
│   │   └── equations.py           # Physics calculations
│   └── visualization/
├── data/
│   └── 01_raw/
│       └── savant.duckdb          # Baseball Savant data
├── docs/
│   ├── data_noise_robustness.md  # Noise filtering guide
│   └── main_entry_point.md       # Main.py documentation
└── tests/
```

---

## Execution Steps

### Step 1: Test Individual Modules (Optional)

```bash
# Test data loader
python src/data_pipeline/data_loader.py

# Test strategy engine
python src/game_theory/engine.py

# Test context encoder
python src/game_theory/context_encoder.py
```

### Step 2: Run Main Simulation

```bash
# Execute the entry point
python src/main.py
```

### Step 3: Check Output

```bash
# View generated files
ls -lh simulation_result.png aegis_simulation.log

# View simulation log
cat aegis_simulation.log
```

---

## Expected Output

### Console Output

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
   Previous Pitch: FF @ 97.0 mph

🎯 Batter Profile:
   Chase Rate: 32.0%
   ISO: 0.350 (⚠️  HIGH POWER)

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

📝 STRATEGIC RATIONALE:
변화구 Sinker(15%)로, 직전 Four-Seam Fastball(FF) 이후, EV 차이가 +7.9mph로 크며...
```

### Generated Files

1. **simulation_result.png** (46 KB)

   - Trajectory visualization placeholder

2. **aegis_simulation.log**
   - Detailed execution log with timestamps

---

## Command Reference

### Basic Commands

```bash
# Run main simulation
python src/main.py

# Run with Python 3 explicitly
python3 src/main.py

# Run from project root
cd /path/to/aegis-pitching-engine && python src/main.py

# Run in background (macOS/Linux)
nohup python src/main.py > output.log 2>&1 &
```

### Testing Commands

```bash
# Test all modules
python -m pytest tests/

# Test specific module
python src/game_theory/engine.py

# Check for errors
python -m py_compile src/main.py
```

### Data Management

```bash
# Check database
python -c "from src.data_pipeline.data_loader import AegisDataLoader; \
           with AegisDataLoader() as loader: print(loader.get_table_info())"

# Load pitcher data
python -c "from src.data_pipeline.data_loader import AegisDataLoader; \
           with AegisDataLoader() as loader: \
           df = loader.load_pitcher_data(621111); print(f'{len(df)} pitches')"
```

---

## Configuration

### Key Parameters (src/common/config.py)

```python
# Data Noise Filtering
MIN_PITCH_USAGE_THRESHOLD = 0.03  # 3% threshold for ghost pitches
MIN_SAMPLE_SIZE_THRESHOLD = 10     # Minimum samples for Stuff+
LOW_SAMPLE_PENALTY = 0.7           # Penalty for low-sample pitches

# Decision Making
HIGH_LEVERAGE_TEMP = 0.7           # Conservative in critical situations
LOW_LEVERAGE_TEMP = 1.5            # Exploratory in comfortable situations

# Feature Weights
FEATURE_WEIGHTS = {
    'tunneling': 0.30,    # Pitch sequencing similarity
    'ev_delta': 0.20,     # Effective velocity difference
    'chase_rate': 0.15,   # Batter's chase tendency
    'stuff_quality': 0.20,# Pitch quality (Stuff+)
    'command': 0.10,      # Zone command success rate
    'entropy': 0.05       # Pattern unpredictability
}
```

### Modify Configuration

Edit `src/common/config.py` and re-run:

```bash
python src/main.py
```

---

## Troubleshooting

### Issue: Module Not Found

```
ModuleNotFoundError: No module named 'src'
```

**Solution**: Run from project root:

```bash
cd /Users/ekim56/Desktop/aegis-pitching-engine
python src/main.py
```

### Issue: DuckDB File Not Found

```
FileNotFoundError: DuckDB 파일을 찾을 수 없습니다
```

**Solution**: Check database path in config:

```python
# src/common/config.py
DB_PATH = Path(__file__).parent.parent.parent / "data/01_raw/savant.duckdb"
```

### Issue: Import Errors

```
ImportError: cannot import name 'AegisStrategyEngine'
```

**Solution**: Verify all files exist:

```bash
ls -l src/game_theory/engine.py
ls -l src/data_pipeline/data_loader.py
```

### Issue: PyTorch Not Available

```
RuntimeError: Attempting to deserialize object on a CUDA device
```

**Solution**: Force CPU device:

```python
# In main.py
engine = AegisStrategyEngine(device='cpu')
```

---

## Performance Benchmarks

### Execution Time

- **Data Loading**: 0.1s (15,419 pitches from DuckDB)
- **Engine Initialization**: 0.5s (load all sub-modules)
- **Decision Making**: 0.01s (single inference)
- **Total Runtime**: < 1 second

### Memory Usage

- **Peak Memory**: ~200 MB (includes PyTorch models)
- **DuckDB Connection**: Read-only (minimal overhead)

### Scalability

- **Single Decision**: < 1s
- **Full Game (300 pitches)**: ~5s (estimated)
- **Full Season (30,000 pitches)**: ~500s (estimated)

---

## Advanced Usage

### Custom Scenario

```python
# Edit src/main.py

# Example: Change to low-leverage situation
game_state = {
    'outs': 0,
    'count': '1-1',
    'runners': [0, 0, 0],  # Empty bases
    'score_diff': 5,        # 5 run lead
    'inning': 3             # 3rd inning
}
```

### Different Pitcher

```python
# Load data for Justin Verlander (ID: 543037)
pitcher_id = 543037
pitcher_stats = load_pitcher_stats(loader, pitcher_id, year=2024)
```

### Batch Simulation

```python
# Run 100 simulations with different random seeds
for i in range(100):
    np.random.seed(i)
    result = engine.decide_pitch(...)
    print(f"Sim {i}: {result.selected_action.pitch_type}")
```

---

## Development Workflow

### 1. Modify Code

```bash
# Edit any module
vim src/game_theory/engine.py
```

### 2. Test Locally

```bash
# Test the modified module
python src/game_theory/engine.py
```

### 3. Run Full Simulation

```bash
# Execute main entry point
python src/main.py
```

### 4. Validate Output

```bash
# Check results
cat aegis_simulation.log
open simulation_result.png
```

---

## Next Steps

### Immediate

1. ✅ Run `python src/main.py` successfully
2. ✅ Verify output files generated
3. ✅ Understand the AI recommendation

### Short-Term

- [ ] Load different pitcher data (e.g., Gerrit Cole)
- [ ] Modify scenario to low-leverage situation
- [ ] Experiment with different batter profiles

### Long-Term

- [ ] Implement full game simulation (9 innings)
- [ ] Add real-time trajectory visualization
- [ ] Create FastAPI endpoint for web service
- [ ] Fine-tune models on historical outcomes

---

## Resources

### Documentation

- [Main Entry Point Guide](main_entry_point.md)
- [Data Noise Robustness](data_noise_robustness.md)
- [Architecture Overview](architecture.md)

### Data Sources

- **Baseball Savant**: https://baseballsavant.mlb.com/
- **Statcast Search**: https://baseballsavant.mlb.com/statcast_search

### External References

- **Stuff+ Metric**: https://library.fangraphs.com/stuff-plus/
- **Game Theory in Baseball**: https://www.fangraphs.com/tht/game-theory-in-baseball/

---

## Support

### Getting Help

1. Check logs: `cat aegis_simulation.log`
2. Review documentation in `docs/`
3. Test individual modules
4. Verify configuration in `src/common/config.py`

### Common Issues

- Path errors → Run from project root
- Import errors → Check `__init__.py` files
- Data errors → Verify DuckDB file exists
- Device errors → Force CPU with `device='cpu'`

---

**Version**: 1.0.0
**Last Updated**: 2026-01-06
**Status**: ✅ Production Ready
