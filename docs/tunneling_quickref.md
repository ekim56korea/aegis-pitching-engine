# TunnelingAnalyzer - Quick Reference Card

## 🚀 Quick Start (5 Lines)

```python
from src.game_theory import TunnelingAnalyzer
from src.data_pipeline import AegisDataLoader

# Load data and get pitcher
with AegisDataLoader() as loader:
    df = loader.load_data_by_year(year=2024, limit=100)
fastball = df[df['pitch_type'] == 'FF'].iloc[0]

# Analyze tunneling
analyzer = TunnelingAnalyzer()
result = analyzer.simulate_counterfactual(
    actual_pitch_data=fastball,
    target_pitch_type='SL',
    pitcher_id=int(fastball['pitcher'])
)

# Visualize
analyzer.visualize_tunneling(result, save_path='tunneling.png')
```

---

## 📊 Key Methods

### 1. get_pitch_profile

```python
profile = analyzer.get_pitch_profile(
    pitcher_id=621111,
    pitch_type='FF'
)
# Returns: {release_pos, release_vel, spin_rate, spin_axis, avg_plate_speed}
```

### 2. simulate_counterfactual

```python
result = analyzer.simulate_counterfactual(
    actual_pitch_data=row,
    target_pitch_type='SL',
    pitcher_id=621111  # Optional
)
# Returns: {actual_traj, cf_traj, actual_vaa, cf_vaa, ...}
```

### 3. calculate_tunnel_score

```python
score_info = analyzer.calculate_tunnel_score(
    traj1=result['actual_traj'],
    time1=result['actual_time'],
    traj2=result['cf_traj'],
    time2=result['cf_time']
)
# Returns: {tunnel_score, distance_at_decision, ...}
```

### 4. calculate_approach_angles

```python
angles = analyzer.calculate_approach_angles(trajectory)
# Returns: {vaa: -8.62, haa: 3.06}
```

---

## 🎯 Pitch Type Codes

| Code | Name      | Avg RPM | Avg Speed |
| ---- | --------- | ------- | --------- |
| FF   | 4-Seam FB | 2300    | 100%      |
| SI   | Sinker    | 2150    | 98%       |
| FC   | Cutter    | 2400    | 96%       |
| SL   | Slider    | 2500    | 90%       |
| CU   | Curveball | 2650    | 83%       |
| CH   | Changeup  | 1800    | 88%       |

---

## 📈 Interpretation Guide

### Tunnel Score

- **> 0.9**: Elite tunneling (SI/CH)
- **0.8 - 0.9**: Excellent (SL/FC)
- **0.6 - 0.8**: Good
- **< 0.6**: Poor

### VAA (Vertical Approach Angle)

- **-5° to -7°**: Rising FB perception
- **-8° to -10°**: Typical FB/SL
- **-12° to -15°**: Breaking ball (CU)

### Decision Point Distance

- **< 0.1m**: Perfect deception
- **0.1 - 0.3m**: Strong tunneling
- **0.3 - 0.5m**: Moderate
- **> 0.5m**: Weak

---

## 🔧 Configuration

```python
# Custom physics engine
from src.physics_engine import SavantPhysicsEngine

engine = SavantPhysicsEngine(
    temperature_f=85.0,      # Hot day
    pressure_hg=29.0,        # Low pressure
    humidity_percent=80.0,   # Humid
    elevation_ft=5280.0      # Denver (Coors Field)
)

analyzer = TunnelingAnalyzer(
    physics_engine=engine,
    dt=0.0005  # Finer time step
)
```

---

## 🐛 Troubleshooting

### Issue: "No data found for pitcher_id"

**Solution**: Check if pitcher exists in database

```python
with AegisDataLoader() as loader:
    df = loader.load_pitcher_data(621111)
    print(f"Found {len(df)} pitches")
```

### Issue: "Unknown pitch type"

**Solution**: Use standard codes (FF, SI, FC, SL, CU, CH)

### Issue: "Connection already closed"

**Solution**: Use context manager or create new loader

```python
# DON'T: Reuse loader
loader = AegisDataLoader()
df1 = loader.load_data(...)
df2 = loader.load_data(...)  # ❌ Error

# DO: Use context manager
with AegisDataLoader() as loader:
    df = loader.load_data(...)  # ✅ OK
```

---

## 📊 Batch Analysis Example

```python
# Compare all pitch types
pitcher_id = 621111
fastball = df[df['pitch_type'] == 'FF'].iloc[0]
target_types = ['SI', 'FC', 'SL', 'CU', 'CH']

scores = []
for target in target_types:
    result = analyzer.simulate_counterfactual(
        fastball, target, pitcher_id
    )
    tunnel_info = analyzer.calculate_tunnel_score(
        result['actual_traj'], result['actual_time'],
        result['cf_traj'], result['cf_time']
    )
    scores.append({
        'target': target,
        'score': tunnel_info['tunnel_score'],
        'vaa': result['cf_vaa']
    })

# Find best combination
best = max(scores, key=lambda x: x['score'])
print(f"Best: FF → {best['target']} (Score: {best['score']:.3f})")
```

---

## 💡 Pro Tips

1. **Use Real Pitcher Data**: Always provide `pitcher_id` for realistic profiles
2. **Cache Profiles**: Store `get_pitch_profile()` results to avoid repeated DB queries
3. **Visualize Everything**: VAA/HAA display helps validate physics
4. **Check Decision Point**: Verify 0.167s timing is appropriate for analysis
5. **Batch Process**: Process multiple pitchers/types in one session

---

## 📚 Quick Links

- **Full Documentation**: [docs/tunneling_production.md](tunneling_production.md)
- **Source Code**: [src/game_theory/tunneling.py](../src/game_theory/tunneling.py)
- **Physics Engine**: [src/physics_engine/savant_physics.py](../src/physics_engine/savant_physics.py)
- **Data Loader**: [src/data_pipeline/data_loader.py](../src/data_pipeline/data_loader.py)

---

**Last Updated**: 2026-01-06
**Version**: 1.0.0 Production
