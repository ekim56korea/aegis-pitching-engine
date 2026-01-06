# Aegis Strategy Room - Interactive 3D Dashboard

## Overview

The **Aegis Strategy Room** is an interactive Streamlit dashboard that visualizes pitching strategy in real-time with 3D trajectory plots powered by Plotly.

---

## Features

### 1. **Control Tower (Sidebar)**

- ⚙️ **Game State**: Inning, score, outs, runners
- ⚾ **Count**: Balls and strikes
- 🎯 **Pitcher**: Pitch count, role, handedness
- 🎯 **Matchup**: Batter profile (chase rate, power, etc.)
- 📊 **Pitcher Selection**: Choose from real MLB pitchers

### 2. **Situation & Recommendation**

- 📋 **Real-time situation summary**
- 🤖 **AI recommendation** with large visual display
- 📊 **Top 3 action probabilities** with progress bars
- 📝 **Strategic rationale** in natural language

### 3. **3D Trajectory Visualization**

- 🎬 **Interactive Plotly 3D plot**
- Previous pitch (gray dashed line)
- Selected pitch (colored solid line)
- Tunnel point marker at 0.167s
- Strike zone visualization
- Mouse controls (rotate, zoom, pan)

### 4. **Physics Metrics Table**

- 📊 Expected metrics for selected pitch
- Tunnel score, EV delta, command risk
- Data quality indicators
- Filtered pitch information

---

## Installation

### Dependencies

```bash
# Install required packages
pip install streamlit plotly pandas numpy

# Or via poetry
poetry add streamlit plotly
```

**Required modules:**

- `streamlit >= 1.28.0`
- `plotly >= 5.17.0`
- `pandas >= 2.0.0`
- `numpy >= 1.24.0`

---

## Usage

### Quick Start

```bash
# Navigate to project root
cd /Users/ekim56/Desktop/aegis-pitching-engine

# Run the dashboard
streamlit run src/dashboard/app.py
```

### Command Options

```bash
# Run on specific port
streamlit run src/dashboard/app.py --server.port 8501

# Run with auto-reload
streamlit run src/dashboard/app.py --server.runOnSave true

# Run in headless mode (server)
streamlit run src/dashboard/app.py --server.headless true
```

---

## User Interface

### Control Tower (Sidebar)

```
⚙️ Control Tower
├── 🏟️ Game State
│   ├── Inning: 1-9 (slider)
│   ├── Score Diff: -5 to +5 (slider)
│   ├── Outs: 0, 1, 2 (select)
│   └── Runners: 1st, 2nd, 3rd (checkboxes)
├── ⚾ Count
│   ├── Balls: 0-3 (select)
│   └── Strikes: 0-2 (select)
├── 🎯 Pitcher
│   ├── Pitch Count: 0-120 (slider)
│   ├── Role: SP/RP (select)
│   ├── Hand: R/L (select)
│   ├── Previous Pitch: FF, SL, etc. (select)
│   └── Previous Velocity: 85-105 mph (input)
├── 🎯 Matchup
│   ├── Batter Hand: R/L (select)
│   ├── Chase Rate: 0-50% (slider)
│   ├── Whiff Rate: 0-50% (slider)
│   ├── ISO: 0-0.500 (slider)
│   └── OPS: 0.5-1.5 (slider)
├── 📊 Pitcher Selection
│   └── Select: Walker Buehler, Default (select)
└── 🚀 Simulate Strategy (button)
```

### Main Dashboard Layout

```
┌─────────────────────────────────────────────────────────────┐
│ ⚾ Aegis Strategy Room - 3D Interactive Dashboard           │
├──────────────────────┬──────────────────────────────────────┤
│ 📋 Situation Summary │ 🤖 AI Recommendation                 │
│  - Inning/Outs       │  ┌────────────────────┐              │
│  - Count             │  │   SI @ shadow_out  │              │
│  - Score             │  │   (Large Display)  │              │
│  - Leverage          │  └────────────────────┘              │
│                      │  📍 Location: (0.58, 2.00)           │
│ 🎯 Batter Profile    │  📊 Top 3 Probabilities:             │
│  - Hand: L           │   1. SI_chase_out: 1.8%             │
│  - Chase: 32%        │   2. CH_chase_out: 1.7%             │
│  - ISO: 0.350        │   3. CH_chase_low: 1.7%             │
├──────────────────────┴──────────────────────────────────────┤
│ 📝 Strategic Rationale:                                     │
│ "변화구 Sinker(15%)로, 직전 Four-Seam Fastball(FF) 이후..." │
├─────────────────────────────────────────────────────────────┤
│ 🎬 3D Pitch Trajectory                                      │
│ [Interactive Plotly 3D Plot]                                │
│  - Previous pitch (gray dashed)                             │
│  - Selected pitch (colored solid)                           │
│  - Tunnel point markers                                     │
│  - Strike zone                                              │
├─────────────────────────────────────────────────────────────┤
│ 📊 Expected Physics Metrics                                 │
│ [Data Table]                                                │
│  Metric              │ Value                                │
│  Pitch Type          │ SI                                   │
│  Zone                │ shadow_out_low                       │
│  Tunnel Score        │ 0.845                                │
│  ...                 │ ...                                  │
└─────────────────────────────────────────────────────────────┘
```

---

## 3D Visualization

### Trajectory Plot Features

**Elements:**

- 🏠 **Home Plate**: White square at (0, 0, 0)
- 🏔️ **Mound**: Brown circle at (0, 60.5, 0)
- 📈 **Previous Pitch**: Gray dashed line
- 📈 **Selected Pitch**: Colored solid line (pitch-specific)
- 🎯 **Tunnel Point**: Markers at 0.167s
- 🟢 **Strike Zone**: Green rectangle

**Pitch Colors:**

- FF (Four-Seam): Red
- SI (Sinker): Orange
- FC (Cutter): Gold
- SL (Slider): Yellow
- CU (Curveball): Green
- CH (Changeup): Cyan
- KC (Knuckle Curve): Blue
- ST (Sweeper): Purple

**Interactivity:**

- 🖱️ **Rotate**: Click and drag
- 🔍 **Zoom**: Scroll wheel
- 📐 **Pan**: Right-click and drag
- 🏠 **Reset**: Double-click

---

## Performance Optimization

### Caching Strategy

```python
# Resource caching (engines/loaders)
@st.cache_resource
def load_strategy_engine():
    engine = AegisStrategyEngine(device='cpu')
    return engine

# Data caching (pitcher stats)
@st.cache_data
def load_pitcher_data(pitcher_id: int, year: int):
    # ... load data ...
    return pitcher_stats
```

**Benefits:**

- ✅ **Engine loaded once** per session
- ✅ **Data cached** by (pitcher_id, year)
- ✅ **Fast re-runs** after first simulation
- ✅ **Reduced memory usage**

### Performance Metrics

- **First Load**: ~2-3 seconds (engine initialization)
- **Subsequent Runs**: ~0.1 seconds (cached)
- **3D Plot Rendering**: ~0.2 seconds
- **Total Simulation**: < 1 second (after caching)

---

## Example Scenarios

### Scenario 1: High Leverage Crisis

**Setup:**

- Inning: 9th
- Score: +1 (leading)
- Outs: 2
- Runners: Bases loaded
- Count: 3-2
- Pitch Count: 98

**Result:**

- Recommendation: SI @ shadow_out_low
- Rationale: "변화구 Sinker(15%)로 승부처 상황에서 확실한 공 선택"

### Scenario 2: Low Leverage Exploration

**Setup:**

- Inning: 3rd
- Score: +5 (comfortable lead)
- Outs: 0
- Runners: Empty
- Count: 1-1
- Pitch Count: 45

**Result:**

- Recommendation: CU @ shadow_in_mid
- Rationale: "여유 있는 상황으로 다양한 선택 시도"

---

## Customization

### Add New Pitcher

```python
# In render_sidebar()
pitcher_presets = {
    "Walker Buehler (621111)": 621111,
    "Gerrit Cole (543037)": 543037,      # Add this
    "Shohei Ohtani (660271)": 660271,    # Add this
    "Default Stats": None
}
```

### Modify Pitch Colors

```python
# In plot_3d_trajectory()
pitch_colors = {
    'FF': '#FF0000',  # Change to your preferred color
    'SL': '#00FF00',  # Custom green for slider
    # ...
}
```

### Adjust 3D Camera Angle

```python
# In plot_3d_trajectory()
camera=dict(
    eye=dict(x=2.0, y=-2.0, z=1.5)  # Adjust these values
)
```

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'streamlit'"

**Solution:**

```bash
pip install streamlit plotly
```

### Issue: Dashboard not loading

**Solution:**

1. Check you're in project root
2. Verify all dependencies installed
3. Check logs in terminal

### Issue: 3D plot not rendering

**Solution:**

1. Update Plotly: `pip install --upgrade plotly`
2. Clear browser cache
3. Try different browser

### Issue: "FileNotFoundError: DuckDB not found"

**Solution:**
Dashboard uses fallback data automatically. To use real data:

```bash
# Ensure database exists
ls data/01_raw/savant.duckdb
```

---

## Advanced Features

### Real-Time Updates

```python
# Add auto-refresh
st.checkbox("Auto-refresh every 5 seconds")
if st.session_state.get('auto_refresh', False):
    time.sleep(5)
    st.rerun()
```

### Export Results

```python
# Add download button
if st.button("Download Results"):
    df = pd.DataFrame({
        'Pitch': [result.selected_action.pitch_type],
        'Zone': [result.selected_action.zone],
        # ...
    })
    st.download_button(
        "Download CSV",
        df.to_csv(index=False),
        "results.csv"
    )
```

### Multiple Simulations

```python
# Run Monte Carlo
if st.button("Run 100 Simulations"):
    results = []
    for i in range(100):
        result = engine.decide_pitch(...)
        results.append(result.selected_action.pitch_type)

    # Show distribution
    st.bar_chart(pd.Series(results).value_counts())
```

---

## API Reference

### Key Functions

#### `render_sidebar() -> Tuple`

Returns game state, pitcher state, matchup state, pitcher ID, and button state.

#### `plot_3d_trajectory(prev_pitch, selected_pitch, prev_velo, selected_velo) -> go.Figure`

Creates 3D Plotly figure with trajectory visualization.

#### `load_strategy_engine() -> AegisStrategyEngine`

Cached function that returns strategy engine instance.

#### `load_pitcher_data(pitcher_id, year) -> Dict`

Cached function that loads pitcher statistics from database.

---

## Deployment

### Local Development

```bash
streamlit run src/dashboard/app.py
```

### Production Deployment

```bash
# Streamlit Cloud
# 1. Push to GitHub
# 2. Connect to streamlit.io
# 3. Deploy from repo

# Docker
docker run -p 8501:8501 -v $(pwd):/app streamlit:latest
```

### Configuration File

Create `.streamlit/config.toml`:

```toml
[server]
port = 8501
enableCORS = false

[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
```

---

## Resources

- **Streamlit Docs**: https://docs.streamlit.io/
- **Plotly 3D Plots**: https://plotly.com/python/3d-charts/
- **Main Engine**: `src/main.py`
- **Strategy Engine**: `src/game_theory/engine.py`

---

**Version**: 1.0.0
**Last Updated**: 2026-01-06
**Status**: ✅ Production Ready
