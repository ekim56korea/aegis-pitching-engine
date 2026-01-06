# ContextEncoder Changelog

## Version 4.0 (Final) - SP/RP Differentiation + Fatigue Modeling

**Date**: 2024

**Dimensions**: 40 → 42 (+2)

### New Features

1. **Pitcher Role One-Hot Encoding** (+2 dims)
   - Distinguishes between SP (Starter) and RP (Reliever)
   - Enables model to learn role-specific patterns
   - Location: Index 29-30

2. **Role-Specific Fatigue Index** (+1 dim, replaces old pitch_count)
   - **SP**: pitch_count / 100.0 (baseline 100 pitches)
   - **RP**: pitch_count / 30.0 (baseline 30 pitches)
   - Can exceed 1.0 for overwork scenarios
   - Location: Index 36

### Changes

- Removed: Generic pitch_count normalization
- Added: `_encode_pitcher_role()` method
- Added: `_calculate_fatigue()` method with role-aware logic
- Updated: All test cases to include 'role' field
- Updated: Feature breakdown section in tests

### Impact

- **More realistic fatigue modeling**: RP at 35 pitches ≠ SP at 35 pitches
- **Better decision-making**: Model learns different strategies for SP vs RP
- **Minimal overhead**: Only 5% dimension increase for significant improvement

### Test Results

✅ All encodings produce correct 42-dim tensors  
✅ SP fatigue: 65 pitches = 0.650 (normal)  
✅ RP fatigue: 35 pitches = 1.167 (overwork)  
✅ Batch encoding works: [3, 42]  

---

## Version 3.0 - Full Batter Threat Matrix

**Dimensions**: 36 → 40 (+4)

### New Features

1. **Whiff Rate** (+1 dim)
   - Normalized by 0.5
   - Measures strikeout probability

2. **ISO (Isolated Power)** (+1 dim)
   - Normalized by 0.4
   - Measures extra-base hit threat

3. **GB/FB Ratio** (+1 dim)
   - Normalized: (ratio - 0.5) / 2.0
   - Indicates ground ball vs fly ball tendency

4. **OPS** (+1 dim)
   - Normalized: (OPS - 0.5) / 0.6
   - Overall offensive threat level

### Impact

- Comprehensive batter profiling
- Strategic differentiation: contact hitter vs power hitter vs balanced
- Model can learn matchup-specific strategies

---

## Version 2.0 - Game Rules + Enhanced Context

**Dimensions**: 30 → 36 (+6)

### New Features

1. **Outs One-Hot** (+3 dims)
   - 0, 1, 2 outs
   - Critical for leverage situations

2. **Platoon Matchup Binary** (+1 dim)
   - Same-handed = 1.0 (pitcher advantage)
   - Opposite = 0.0 (batter advantage)

3. **Updated Continuous Features** (+2 dims)
   - Pitch count normalization
   - Chase rate (O-Swing%)

### Impact

- Baseball rules properly reflected
- Platoon splits considered
- More accurate game state representation

---

## Version 1.0 - Initial Release

**Dimensions**: 30

### Features

1. **Count One-Hot** (12 dims)
2. **Runners One-Hot** (8 dims)
3. **TTO One-Hot** (4 dims)
4. **Batter Hand One-Hot** (2 dims)
5. **Continuous** (4 dims)
   - Entropy
   - Score difference
   - Inning
   - Previous velocity

### Impact

- Basic game state encoding
- Foundation for neural network input
- Proof of concept

---

## Summary Table

| Version | Dims | Key Addition | Use Case |
|---------|------|--------------|----------|
| v1.0 | 30 | Basic state encoding | Initial implementation |
| v2.0 | 36 | Outs + Platoon matchup | Rule compliance |
| v3.0 | 40 | Full Batter Threat Matrix | Strategic profiling |
| **v4.0** | **42** | **SP/RP + Fatigue** | **Realistic pitcher modeling** |

---

## Future Considerations

Possible v5.0 enhancements:
- Weather conditions (temperature, wind)
- Park factors (dimensions, altitude)
- Recent performance metrics (last 5 games)
- Pitch mix diversity (repertoire size)

**Note**: Each addition must be carefully evaluated for:
- Computational cost
- Data availability
- Marginal benefit
- Model interpretability
