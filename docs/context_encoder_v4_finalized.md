# ContextEncoder v4 - Finalized (42 Dimensions)

## 📋 Overview

**Purpose**: Encode baseball game state into 42-dimensional PyTorch tensor for neural network input

**Version**: v4 (Finalized)

**Key Enhancement**: SP/RP differentiation with role-specific fatigue modeling

---

## 🏗️ Feature Structure (42 Dimensions)

### Categorical Features (32 dims)

1. **Count One-Hot**: 12 dims

   - All 12 count states: 0-0, 0-1, 0-2, 1-0, 1-1, 1-2, 2-0, 2-1, 2-2, 3-0, 3-1, 3-2

2. **Runners One-Hot**: 8 dims

   - All 8 base states: (0,0,0), (1,0,0), (0,1,0), (1,1,0), (0,0,1), (1,0,1), (0,1,1), (1,1,1)

3. **Outs One-Hot**: 3 dims

   - 0, 1, 2 outs

4. **TTO (Times Through Order) One-Hot**: 4 dims

   - 1st time (투수 유리)
   - 2nd time (균형)
   - 3rd time (타자 유리)
   - 4th+ time (clipped)

5. **Batter Hand One-Hot**: 2 dims

   - L, R

6. **Pitcher Role One-Hot**: 2 dims ⭐ **NEW**

   - SP (Starter)
   - RP (Reliever)

7. **Platoon Matchup Binary**: 1 dim
   - Same-handed = 1.0 (투수 유리)
   - Opposite = 0.0 (타자 유리)

### Continuous Features (10 dims)

#### Game Context (4 dims)

8. **Entropy**: Already normalized [0, 1]

   - Pitch sequence randomness from EntropyMonitor

9. **Score Diff (normalized)**: [-1, 1]

   - Clipped to [-5, +5] then normalized

10. **Inning (normalized)**: [0, 1]

    - (inning - 1) / 8.0 (1회 → 0.0, 9회 → 1.0)

11. **Previous Velocity (normalized)**: [0, 1]
    - (velo - 70) / 35.0 (70mph → 0.0, 105mph → 1.0)

#### Fatigue Index (1 dim) ⚡ **NEW**

12. **Fatigue Index**: [0, 1.5+]
    - **SP**: pitch_count / 100.0
      - 0-80: Fresh
      - 80-100: Normal
      - 100-120: Overwork (1.0 ~ 1.2)
    - **RP**: pitch_count / 30.0
      - 0-20: Fresh
      - 20-30: Normal
      - 30-40+: High fatigue (1.0 ~ 1.3+)

#### Batter Threat Matrix (5 dims)

13. **Chase Rate**: [0, 1]

    - O-Swing% (낮을수록 좋은 선구안)
    - 낮음 → 존 공략, 높음 → 유인구 전략

14. **Whiff Rate (normalized)**: [0, 1]

    - Whiff% / 0.5 (헛스윙률)
    - 낮음 → 컨택 좋음, 높음 → 삼진 가능

15. **ISO (normalized)**: [0, 1]

    - ISO / 0.4 (Isolated Power)
    - 낮음 → 단타 타자, 높음 → 장타 위협

16. **GB/FB Ratio (normalized)**: [0, 1]

    - (GB/FB - 0.5) / 2.0
    - 높음 → 땅볼 타자 (병살타 유도), 낮음 → 플라이볼 (홈런 주의)

17. **OPS (normalized)**: [0, 1]
    - (OPS - 0.5) / 0.6
    - Overall threat level

---

## 🆚 SP vs RP Comparison

### Design Philosophy

Starters and relievers have **completely different usage patterns** and fatigue curves:

| Role   | Typical Usage | Pitch Count Range | Fatigue Baseline | Strategy                      |
| ------ | ------------- | ----------------- | ---------------- | ----------------------------- |
| **SP** | 5-7 innings   | 80-110 pitches    | 100 pitches      | Pacing, gradual fatigue       |
| **RP** | 1-2 innings   | 15-35 pitches     | 30 pitches       | Maximum effort, rapid fatigue |

### Fatigue Index Examples

```python
# Starter scenarios
SP at 65 pitches:  0.650 fatigue  # Mid-game, comfortable
SP at 95 pitches:  0.950 fatigue  # Tiring, still capable
SP at 110 pitches: 1.100 fatigue  # Overwork

# Reliever scenarios
RP at 15 pitches:  0.500 fatigue  # Fresh
RP at 28 pitches:  0.933 fatigue  # Normal
RP at 35 pitches:  1.167 fatigue  # ⚠️ High fatigue (overused)
```

### Why This Matters

- A **starter at 80 pitches** = 0.80 fatigue (normal)
- A **reliever at 30 pitches** = 1.00 fatigue (at limit)
- **Same pitch count ≠ Same fatigue!**

The model can now learn:

- SP: Gradual performance decline after 90-100 pitches
- RP: Sharp performance drop after 30 pitches
- Different strategic implications for each role

---

## 📊 Test Results

### Test Case 1: Contact Hitter (SP)

- **Pitcher**: RHP, SP, 65 pitches → Fatigue: 0.650
- **Batter**: LHB, Contact hitter (GB/FB 1.8, Low ISO)
- **Strategy**: Induce ground ball double play

### Test Case 2: Power Hitter (RP, Crisis)

- **Pitcher**: LHP, RP, 35 pitches → Fatigue: 1.167 ⚠️
- **Batter**: RHB, Power hitter (ISO .280, FB hitter)
- **Situation**: 9th inning, bases loaded, 2 outs, full count
- **Strategy**: Chase pitch for strikeout, avoid inside zone

### Test Case 3: Average Batter (SP, Fresh)

- **Pitcher**: RHP, SP, 12 pitches → Fatigue: 0.120
- **Batter**: RHB, Balanced profile
- **Strategy**: Standard repertoire, multiple options

### Batch Encoding

- Successfully created batch tensor: `[3, 42]` ✅
- All features properly concatenated
- No dimension mismatches

---

## 🎯 Usage Example

```python
from game_theory import ContextEncoder

encoder = ContextEncoder(device='cpu')

# Example: Reliever in high-leverage situation
game_state = {
    'outs': 2,
    'count': '3-2',
    'runners': [1, 1, 1],  # Bases loaded
    'score_diff': -1,      # Down by 1
    'inning': 9            # 9th inning
}

pitcher_state = {
    'hand': 'R',
    'role': 'RP',          # ⚠️ Reliever
    'pitch_count': 32,     # 32/30 = 1.067 fatigue
    'entropy': 0.65,
    'prev_pitch': 'SL',
    'prev_velo': 88.5
}

matchup_state = {
    'batter_hand': 'L',
    'times_faced': 1,
    'chase_rate': 0.35,
    'whiff_rate': 0.28,
    'iso': 0.185,
    'gb_fb_ratio': 1.2,
    'ops': 0.780
}

# Encode to 42-dim tensor
encoded = encoder.encode(game_state, pitcher_state, matchup_state)
print(encoded.shape)  # torch.Size([1, 42])
```

---

## 🔄 Version History

| Version | Dims   | Key Features                                      |
| ------- | ------ | ------------------------------------------------- |
| v1      | 30     | Basic count, runners, TTO, entropy, score, inning |
| v2      | 36     | + Outs (3), Platoon (1), updated continuous (2)   |
| v3      | 40     | + Full Batter Threat Matrix (5 dims)              |
| **v4**  | **42** | **+ Pitcher Role (2), Fatigue Index (1)**         |

---

## 💡 Key Improvements in v4

1. **Role-Specific Modeling**

   - SP and RP are fundamentally different
   - Model can learn distinct patterns for each role

2. **Realistic Fatigue Modeling**

   - No more single "pitch_count / 100" for everyone
   - Relative fatigue based on role expectations
   - RP at 35 pitches = more fatigued than SP at 70 pitches

3. **Enhanced Strategic Context**

   - Leverage situations already captured (inning, score, runners, outs)
   - Now combined with role-specific fatigue for better decision-making

4. **Maintained Efficiency**
   - Only 2 additional dimensions (5% increase)
   - Significant improvement in modeling fidelity

---

## ✅ Implementation Status

- [x] Add Pitcher Role One-Hot encoding (2 dims)
- [x] Add Fatigue Index calculation (1 dim)
- [x] Implement `_encode_pitcher_role()` method
- [x] Implement `_calculate_fatigue()` method with SP/RP logic
- [x] Update test cases with 'role' field
- [x] Add fatigue comparison section to tests
- [x] Verify 42-dim output shape
- [x] Test batch encoding with multiple scenarios
- [x] Document SP vs RP differences
- [x] Complete validation: All tests passing ✅

---

## 📝 Notes

- **Fatigue Index can exceed 1.0** for overwork scenarios (expected behavior)
- **SP baseline: 100 pitches** (80-100 normal, 110+ overwork)
- **RP baseline: 30 pitches** (20-30 normal, 35+ rapid decline)
- Model will learn optimal pitch selection based on role + fatigue combination
- Type checking warnings in IDE are cosmetic (code runs perfectly)

---

**Status**: ✅ Finalized and Ready for Production

**Last Updated**: 2024
