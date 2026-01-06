# 🎯 Aegis Strategy Room - User Guide

## Welcome to the Interactive 3D Strategy Dashboard

The Aegis Strategy Room is your command center for visualizing and simulating pitching strategies in real-time. This guide will help you get started.

---

## 🚀 Quick Start (3 Steps)

### Step 1: Install Dependencies

```bash
# Navigate to project directory
cd /Users/ekim56/Desktop/aegis-pitching-engine

# Install dashboard dependencies
pip install -r requirements-dashboard.txt

# Or install manually
pip install streamlit plotly
```

### Step 2: Launch Dashboard

**Option A: Using launcher script (Recommended)**

```bash
./launch_dashboard.sh
```

**Option B: Direct command**

```bash
streamlit run src/dashboard/app.py
```

### Step 3: Open Browser

Dashboard automatically opens at: **http://localhost:8501**

If not, manually navigate to: `http://localhost:8501` in your browser

---

## 🎮 Using the Dashboard

### Control Tower (Left Sidebar)

The sidebar is your control panel for setting up game scenarios:

#### 🏟️ Game State

1. **Inning** (1-9): Current inning of the game
2. **Score Difference** (-5 to +5): Positive = leading, Negative = trailing
3. **Outs** (0-2): Number of outs in the inning
4. **Runners** (checkboxes): Select which bases are occupied

**Example Scenarios:**

- **High Leverage**: 9th inning, +1 score, 2 outs, bases loaded
- **Low Leverage**: 3rd inning, +5 score, 0 outs, bases empty

#### ⚾ Count

- **Balls** (0-3): Current ball count
- **Strikes** (0-2): Current strike count

**Common Counts:**

- `0-0`: Fresh at-bat
- `3-2`: Full count (maximum pressure)
- `0-2`: Pitcher ahead
- `3-0`: Batter ahead

#### 🎯 Pitcher Status

1. **Pitch Count** (0-120): Total pitches thrown
   - 0-60: Fresh
   - 60-90: Normal fatigue
   - 90-120: High fatigue
2. **Role**: SP (Starter) or RP (Reliever)
3. **Hand**: R (Right) or L (Left)
4. **Previous Pitch**: Type of last pitch thrown
5. **Previous Velocity**: Speed of last pitch (mph)

#### 🎯 Matchup Profile

Configure the batter's characteristics:

1. **Batter Hand**: R (Right) or L (Left)
2. **Chase Rate** (0-50%): Tendency to swing at balls outside zone
   - Low (< 25%): Disciplined hitter
   - Medium (25-35%): Average
   - High (> 35%): Aggressive
3. **Whiff Rate** (0-50%): Tendency to miss when swinging
   - Low (< 20%): Strong contact
   - Medium (20-30%): Average
   - High (> 30%): Susceptible to whiffs
4. **ISO** (0-0.500): Isolated power metric
   - Low (< 0.150): Contact hitter
   - Medium (0.150-0.250): Average power
   - High (> 0.250): Power threat
5. **OPS** (0.5-1.5): On-base plus slugging
   - Poor (< 0.700): Below average
   - Good (0.700-0.900): Average to above average
   - Elite (> 0.900): Star player

#### 📊 Pitcher Selection

- **Walker Buehler**: Real 2024 MLB data (15,419 pitches)
- **Default Stats**: Generic pitcher profile

### 🚀 Simulate Button

Once all parameters are set:

1. Click **"🚀 Simulate Strategy"** button
2. Wait ~1-2 seconds for AI to process
3. Results appear in main area

---

## 📊 Understanding the Results

### Top Section: Situation & Recommendation

**Left Column - Situation Summary:**

- Current game state recap
- Leverage level (HIGH/MEDIUM/LOW)
- Entropy status (pattern predictability)
- Batter profile summary

**Right Column - AI Recommendation:**

- **Large display box**: Recommended pitch type and zone
- **Location coordinates**: Exact plate position (X, Z)
- **Confidence level**: Probability of selection
- **Top 3 alternatives**: Other viable options with probabilities

**Example Output:**

```
SI @ shadow_out_low
Location: (0.58, 2.00)
Confidence: 1.8%

Top 3 Probabilities:
1. SI_chase_out: 1.8%
2. CH_chase_out: 1.7%
3. CH_chase_low: 1.7%
```

### Strategic Rationale Box

Natural language explanation of the decision:

**Example:**

> "변화구 Sinker(15%)로, 직전 Four-Seam Fastball(FF) 이후, EV 차이가 +7.9mph로 크며, Sinker(SI)를 shadow_out_low 존에 선택함, (주의: 데이터 신뢰도 50%), 현재 승부처 상황으로 확실한 공을 선택했습니다."

**Key elements explained:**

- **Pitch role**: "주무기" (primary), "보조 구종" (secondary), "변화구" (off-speed)
- **Usage rate**: Percentage in parentheses (e.g., "15%")
- **Sequencing**: Reference to previous pitch
- **Metrics**: EV delta, tunneling, etc.
- **Data quality warning**: Shown if sample size is low
- **Leverage context**: Why this pitch in this situation

---

## 🎬 3D Trajectory Visualization

### Understanding the Plot

**Visual Elements:**

- 🏠 **White Square**: Home plate at (0, 0, 0)
- 🏔️ **Brown Circle**: Pitcher's mound at (0, 60.5, 0)
- 📈 **Gray Dashed Line**: Previous pitch trajectory
- 📈 **Colored Solid Line**: Recommended pitch trajectory
- 🎯 **Tunnel Point Markers**: Position at 0.167 seconds
- 🟢 **Green Rectangle**: Strike zone boundary

**Pitch Type Colors:**

- 🔴 **Red**: FF (Four-Seam Fastball)
- 🟠 **Orange**: SI (Sinker)
- 🟡 **Gold**: FC (Cutter)
- 💛 **Yellow**: SL (Slider)
- 🟢 **Green**: CU (Curveball)
- 🔵 **Cyan**: CH (Changeup)
- 🔷 **Blue**: KC (Knuckle Curve)
- 🟣 **Purple**: ST (Sweeper)

### Interacting with the 3D Plot

**Mouse Controls:**

- **Rotate**: Click and drag
- **Zoom**: Scroll wheel or pinch (trackpad)
- **Pan**: Right-click and drag (or Shift + drag)
- **Reset View**: Double-click anywhere on plot

**What to Look For:**

1. **Tunneling Effect**: Do the trajectories overlap at the tunnel point?
   - Good tunneling = trajectories very close at 0.167s
   - Poor tunneling = trajectories diverge early
2. **Movement Pattern**: Does the pitch break significantly?
   - Fastballs = minimal break
   - Breaking balls = dramatic drop/sweep
3. **Strike Zone Entry**: Does the pitch cross the green zone?
   - Heart of zone = most hittable
   - Shadow/chase = edges and outside

---

## 📊 Physics Metrics Table

Bottom section shows expected metrics for the selected pitch:

| Metric                 | Description                                    |
| ---------------------- | ---------------------------------------------- |
| **Pitch Type**         | Code (FF, SL, etc.)                            |
| **Zone**               | Target location name                           |
| **Plate Location**     | X, Z coordinates                               |
| **Estimated Velocity** | Expected speed (mph)                           |
| **Tunnel Score**       | Similarity to previous pitch (higher = better) |
| **EV Delta**           | Effective velocity difference                  |
| **Command Risk**       | Zone command success rate                      |
| **Stuff Quality**      | Stuff+ rating with sample penalty              |
| **Chase Score**        | Likelihood of inducing chase                   |
| **Entropy Bonus**      | Pattern unpredictability bonus                 |
| **Data Quality**       | Reliability score (0-100%)                     |

**Note:** Some metrics may show "N/A" if not directly available in current DecisionResult.

---

## 🎯 Example Scenarios

### Scenario 1: The Save Situation

**Goal**: Close out the game

**Setup:**

```
Inning: 9th
Score: +1 (one run lead)
Outs: 2
Runners: Bases loaded
Count: 3-2 (full count)
Pitch Count: 98 (high fatigue)
Batter: L, Chase 32%, Whiff 28%, ISO 0.350 (dangerous)
```

**Expected Recommendation:**

- Conservative pitch (FF or SI)
- Zone: Chase/shadow (avoid heart)
- High confidence in primary pitch
- Rationale emphasizes leverage

**What to Check:**

- Is the pitch going to a safe zone?
- Does it utilize pitcher's best stuff?
- Is tunneling good off previous pitch?

### Scenario 2: The Comfortable Lead

**Goal**: Attack zone, save bullpen

**Setup:**

```
Inning: 3rd
Score: +5 (five run lead)
Outs: 0
Runners: Empty
Count: 1-1
Pitch Count: 45 (fresh)
Batter: R, Chase 40%, Whiff 30%, ISO 0.180 (average)
```

**Expected Recommendation:**

- Exploratory pitch (CU, CH, or secondary)
- Zone: Shadow/heart (challenge batter)
- Lower confidence (more options)
- Rationale mentions exploration

**What to Check:**

- Is the AI using secondary pitches?
- Higher entropy/unpredictability?
- More aggressive zone targeting?

### Scenario 3: First Pitch Strike

**Goal**: Get ahead in count

**Setup:**

```
Inning: 5th
Score: 0 (tied)
Outs: 1
Runners: Runner on 1st
Count: 0-0 (fresh at-bat)
Pitch Count: 65 (moderate)
Batter: R, Chase 28%, Whiff 25%, ISO 0.200
```

**Expected Recommendation:**

- Primary fastball (FF or SI)
- Zone: Shadow (steal strike)
- High probability on fastball
- Rationale mentions getting ahead

---

## 🔧 Advanced Features

### Caching System

The dashboard uses Streamlit's caching for performance:

**What's Cached:**

- ✅ Strategy Engine (loaded once per session)
- ✅ Data Loader (shared across simulations)
- ✅ Pitcher Data (cached by pitcher ID)

**Benefits:**

- First simulation: ~2 seconds
- Subsequent simulations: ~0.1 seconds
- No redundant data loading

**Cache Clearing:**
If you need to force reload:

1. Press `C` in browser (Streamlit shortcut)
2. Or restart the dashboard

### URL State Persistence

Streamlit automatically saves your inputs between sessions. Refreshing the page preserves:

- Last used game state
- Previous simulation results
- Sidebar settings

### Exporting Results

Currently not implemented, but can be added:

```python
# Add download button
if st.button("Download Results"):
    results_df = pd.DataFrame({
        'Pitch': [result.selected_action.pitch_type],
        'Zone': [result.selected_action.zone],
        'Confidence': [list(result.action_probs.values())[0]],
        'Rationale': [result.rationale]
    })
    st.download_button(
        "Download CSV",
        results_df.to_csv(index=False),
        "aegis_results.csv"
    )
```

---

## 🐛 Troubleshooting

### Dashboard Won't Start

**Issue**: `streamlit: command not found`

**Solution:**

```bash
pip install streamlit plotly
```

---

### 3D Plot Not Rendering

**Issue**: Plot shows blank or error

**Solutions:**

1. Update Plotly: `pip install --upgrade plotly`
2. Clear browser cache: Ctrl+Shift+Delete
3. Try different browser (Chrome recommended)
4. Check browser console (F12) for JavaScript errors

---

### "ModuleNotFoundError" Errors

**Issue**: Can't find `src.game_theory.engine`

**Solution:**
Ensure you're running from project root:

```bash
# Check current directory
pwd  # Should show: /Users/ekim56/Desktop/aegis-pitching-engine

# If not, navigate to root
cd /Users/ekim56/Desktop/aegis-pitching-engine

# Then launch
streamlit run src/dashboard/app.py
```

---

### Slow Performance

**Issue**: Simulations take > 5 seconds

**Solutions:**

1. **First run is slow** - This is normal (loading engines)
2. **Subsequent runs should be fast** - If not, check CPU usage
3. **Clear cache**: Press `C` in browser
4. **Reduce complexity**: Use default pitcher instead of real data

---

### Database Warning

**Issue**: "DuckDB file not found"

**This is OK!** Dashboard automatically uses default stats.

**To use real data:**

1. Ensure database exists: `ls data/01_raw/savant.duckdb`
2. If missing, obtain Baseball Savant data
3. Dashboard will auto-detect and use it

---

## 💡 Tips & Best Practices

### Getting Realistic Results

1. **Use realistic combinations**:

   - High leverage = late innings, close score, runners on base
   - Low leverage = early innings, big lead, bases empty

2. **Fatigue matters**:

   - 0-60 pitches: Use full arsenal
   - 60-90 pitches: Expect reduced command
   - 90+ pitches: Expect conservative choices

3. **Matchup dynamics**:
   - R pitcher vs L batter = platoon advantage
   - High ISO batter + runners on = avoid mistakes
   - High chase rate = use chase zones

### Reading AI Decisions

**When AI recommends fastball:**

- Usually in high leverage
- Batter has low chase rate (respect)
- Previous pitch was off-speed (velocity contrast)

**When AI recommends off-speed:**

- Low leverage (exploration)
- Batter has high chase rate
- Previous pitch was fastball (tunneling)

**When AI uses shadow zones:**

- Trying to steal strikes
- Batter has poor zone coverage
- Ahead in count

**When AI uses chase zones:**

- Behind in count (need swing-and-miss)
- Batter has high chase rate
- Previous pitch set up sequence

### Experimentation Ideas

1. **Same situation, different batters**: How does ISO change recommendations?
2. **Fatigue impact**: Compare pitch 30 vs pitch 100
3. **Leverage scaling**: Same count, different scores
4. **Tunnel testing**: Keep previous pitch same, vary current count
5. **Arsenal exploration**: Switch between pitchers

---

## 📚 Additional Resources

### Project Documentation

- [Main Entry Point](../docs/main_entry_point.md)
- [Data Noise Robustness](../docs/data_noise_robustness.md)
- [Architecture Overview](../docs/architecture.md)
- [Quick Start Guide](../QUICKSTART.md)

### External References

- **Streamlit Docs**: https://docs.streamlit.io/
- **Plotly 3D**: https://plotly.com/python/3d-scatter-plots/
- **Baseball Savant**: https://baseballsavant.mlb.com/

### Video Tutorials

(To be added)

---

## 🚀 Next Steps

1. ✅ Launch dashboard and explore interface
2. ✅ Try different scenarios (high/low leverage)
3. ✅ Experiment with batter profiles
4. ✅ Observe tunneling effects in 3D plot
5. ✅ Compare recommendations across pitchers

---

## 📞 Support

**Issues?**

- Check logs in terminal where dashboard is running
- Review Streamlit error messages in browser
- Verify all dependencies installed
- Ensure running from project root

**Feature Requests?**
Add to project roadmap or implement yourself!

---

**Version**: 1.0.0
**Last Updated**: 2026-01-06
**Status**: ✅ Production Ready

---

**Enjoy the Aegis Strategy Room! ⚾🎯**
