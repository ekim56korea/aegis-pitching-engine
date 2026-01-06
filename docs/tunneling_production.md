# Tunneling Analysis - Production Version

## 📅 Production Release

**Date**: 2026년 1월 6일
**Version**: 1.0.0 (Final Production)

## 🎯 Executive Summary

**TunnelingAnalyzer**는 MLB 투수의 투구 터널링(Pitch Tunneling) 효과를 분석하는 최종 프로덕션 도구입니다. 실제 Statcast 데이터에서 추출한 투수별 구종별 DNA를 기반으로 Delta Injection 방식의 반사실적 시뮬레이션을 수행하며, VAA/HAA 등 고급 메트릭을 계산합니다.

---

## 🏗️ Architecture Overview

### Data Flow

```
DuckDB (7.8M pitches)
    ↓
AegisDataLoader
    ↓
get_pitch_profile (투수별 구종별 평균 DNA)
    ↓
simulate_counterfactual (Delta Injection)
    ↓
SavantPhysicsEngine (Alan Nathan Model)
    ↓
calculate_approach_angles (VAA, HAA)
    ↓
Visualization + Tunnel Score
```

---

## 🔑 Key Features

### 1. Data Integration (AegisDataLoader)

**Purpose**: DuckDB에서 투수별 실제 Statcast 데이터 로드

**Database**:

- 7,799,523 pitch records
- 24 required columns (Statcast standard)
- Years: 2015-2024

**Query Capability**:

```python
loader = AegisDataLoader()
with loader as context:
    df = context.load_pitcher_data(pitcher_id=621111)  # 특정 투수
    df = context.load_data_by_year(year=2024, limit=500)  # 연도별
```

---

### 2. get_pitch_profile - The DNA of the Pitch

**Objective**: 투수별 구종별 평균 특성 추출

**Method Signature**:

```python
def get_pitch_profile(
    self,
    pitcher_id: int,
    pitch_type: str
) -> Dict[str, np.ndarray]:
```

**Output Structure**:

```python
{
    # Kinematics (운동학)
    'release_pos': np.array([x, y, z]),  # Extension(y) 포함 필수
    'release_vel': np.array([vx, vy, vz]),  # Launch Angle 내포

    # Dynamics (동역학)
    'spin_rate': float,  # RPM
    'spin_axis': float,  # Degree (0-360)

    # Validation (검증용)
    'avg_plate_speed': float  # mph (종속 확인)
}
```

**Example Output** (Pitcher 621111, FF):

```
Position: [-0.288, 16.502, 1.771] meters
Velocity: [1.063, -42.529, -1.774] m/s
Spin Rate: 2424 RPM
Spin Axis: 349.7°  # near-backspin
```

**Spin Axis Convention**:

- 0° = Pure Backspin (+x direction)
- 90° = Pure Sidespin (+z direction)
- 180° = Pure Topspin (-x direction)
- 270° = Pure Sidespin (-z direction)

---

### 3. simulate_counterfactual - Delta Injection Method

**Objective**: 실제 투구와 동일한 타이밍/컨디션에서 구종만 변경

**Core Algorithm**:

1. **Profile 추출**:

   ```python
   actual_profile = get_pitch_profile(pitcher_id, 'FF')
   target_profile = get_pitch_profile(pitcher_id, 'SL')
   ```

2. **Delta 계산**:

   ```
   ΔPos = Profile_Target.pos - Profile_Actual.pos
   ΔVel = Profile_Target.vel - Profile_Actual.vel
   ΔSpin = Profile_Target.spin - Profile_Actual.spin
   ```

3. **주입 (Injection)**:

   ```
   Counterfactual_State = Actual_State + Δ
   ```

4. **물리 시뮬레이션**:
   ```python
   cf_time, cf_traj = _simulate_trajectory(cf_state, cf_spin)
   ```

**Example Delta** (FF → SL, Pitcher 621111):

```
ΔPos: [-0.087, 0.110, -0.039] m
ΔVel: [-0.217, 3.846, 0.832] m/s
ΔSpin: [23.33, 0.0, 139.51] rad/s
```

**Physics Engine**:

- Alan Nathan Model (spin saturation)
- Euler integration (dt=0.001s)
- Air density: ρ(T, P, RH, elevation)

---

### 4. calculate_approach_angles - Advanced Metrics

**Objective**: 홈플레이트에서의 접근 각도 계산

**Method**:

```python
def calculate_approach_angles(
    self,
    trajectory: np.ndarray  # [N, 6]
) -> Dict[str, float]:
```

**Formulas**:

**VAA (Vertical Approach Angle)**:

```
VAA = arctan(v_fz / v_fy)  [도 단위]
```

- 음수: 하강 (typical for most pitches)
- 양수: 상승 (rare, rising fastball illusion)

**HAA (Horizontal Approach Angle)**:

```
HAA = arctan(v_fx / v_fy)  [도 단위]
```

- 음수: 좌측으로 이동 (투수 시점)
- 양수: 우측으로 이동 (투수 시점)

**Example Output** (Pitcher 621111):

```
FF: VAA=-8.62°, HAA=3.06°
SL: VAA=-8.88°, HAA=3.10°
```

**Physical Interpretation**:

- VAA ≈ -8~-10°: Typical MLB fastball/slider
- VAA ≈ -5~-7°: Rising fastball perception
- VAA ≈ -12~-15°: Breaking ball (curveball)

---

### 5. Tunnel Score Calculation

**Decision Point**: t = 0.167 seconds

- 투구 후 약 23.8 feet (7.25 meters)
- 타자의 마지막 의사결정 시점

**Formula**:

```
Distance = ||Position_1 - Position_2|| (3D Euclidean)
Tunnel_Score = 1 / (1 + Distance)
```

**Interpretation**:

- Score = 1.0: Perfect tunneling (동일 궤적)
- Score > 0.8: Excellent tunneling
- Score > 0.6: Good tunneling
- Score < 0.5: Poor tunneling

---

## 📊 Results (Real Data - Pitcher 621111)

### Best Tunneling Combinations

| Combo   | Score | Distance | VAA_Actual | VAA_CF | Notes                   |
| ------- | ----- | -------- | ---------- | ------ | ----------------------- |
| FF → SI | 0.914 | 0.095 m  | -8.62°     | -8.30° | ⭐ 최고 조합            |
| FF → CH | 0.897 | 0.114 m  | -8.62°     | -8.56° | Excellent deception     |
| FF → SL | 0.812 | 0.232 m  | -8.62°     | -8.88° | Good tunneling          |
| FF → FC | 0.803 | 0.246 m  | -8.62°     | -8.55° | Solid                   |
| FF → CU | 0.507 | 0.973 m  | -8.62°     | -7.76° | Poor (CU data fallback) |

### Key Insights

1. **Sinker (SI) tunnels best with Fastball**:

   - Smallest position delta at Decision Point
   - Similar vertical approach angles (-8.62° vs -8.30°)
   - Small velocity difference maintains deception

2. **Changeup (CH) also excellent**:

   - Score 0.897 despite velocity reduction
   - VAA nearly identical to FF
   - Arm action similarity critical

3. **Curveball (CU) poor performance**:
   - Used fallback profile (no CU data for this pitcher)
   - Large trajectory deviation
   - Demonstrates importance of real pitcher DNA

---

## 🎨 Visualization

### Output

- **File**: `examples/tunneling_analysis.png`
- **Format**: 2-panel comparison

### Panel 1: Side View (Y-Z Plane)

- X-axis: Distance from Home Plate (m)
- Y-axis: Height (m)
- Features:
  - Actual trajectory (blue solid)
  - Counterfactual trajectory (red dashed)
  - Decision Point markers
  - Strike zone overlay

### Panel 2: Batter's View (X-Z Plane)

- X-axis: Horizontal Position (m)
- Y-axis: Height (m)
- Features:
  - Same trajectory overlays
  - Strike zone box (17 inches × 2 feet)
  - Decision Point markers

### Title Display

```
Tunnel Score: 0.812 | Distance: 0.232m |
VAA: FF=-8.62° / SL=-8.88°
```

### Bottom Info Box

```
Approach Angles:
  FF: VAA=-8.62°, HAA=3.06°
  SL: VAA=-8.88°, HAA=3.10°
```

---

## 🔬 Technical Implementation

### Class Initialization

```python
analyzer = TunnelingAnalyzer(
    data_loader=None,  # Optional, creates new instances as needed
    physics_engine=None,  # Optional, standard conditions default
    dt=0.001  # Time step for Euler integration
)
```

### Full Workflow

```python
# 1. Load data
with AegisDataLoader() as loader:
    df = loader.load_data_by_year(year=2024, limit=500)

fastball_data = df[df['pitch_type'] == 'FF'].iloc[0]
pitcher_id = int(fastball_data['pitcher'])

# 2. Initialize analyzer
analyzer = TunnelingAnalyzer()

# 3. Simulate counterfactual
result = analyzer.simulate_counterfactual(
    actual_pitch_data=fastball_data,
    target_pitch_type='SL',
    pitcher_id=pitcher_id
)

# 4. Calculate tunnel score
tunnel_info = analyzer.calculate_tunnel_score(
    result['actual_traj'], result['actual_time'],
    result['cf_traj'], result['cf_time']
)

# 5. Visualize
analyzer.visualize_tunneling(
    result,
    save_path='examples/tunneling_analysis.png'
)
```

---

## 🧪 Validation

### Physics Verification

✅ **Coordinate System**: 모든 테스트 통과
✅ **Drag Force Direction**: vy < 0 → Fy > 0 확인
✅ **VAA Range**: MLB 평균 범위 내 (-5° ~ -12°)
✅ **Tunnel Score Consistency**: 반복 실행 시 동일 결과

### Data Validation

- **Pitcher 621111**: 15,419 pitches loaded
- **Pitch Types**: FF (2424 RPM), SI (2251 RPM), FC (2561 RPM), SL (2759 RPM), CH (1566 RPM)
- **CU**: No data (fallback profile used)

---

## 📈 Performance

### Computation Time (M1 Mac, CPU)

- Profile extraction: ~0.5s per query
- Trajectory simulation: ~10ms per pitch
- Full comparison (5 types): ~3s total

### Memory Usage

- DuckDB connection: ~50 MB
- Trajectory storage: ~5 KB per simulation
- Peak memory: <200 MB

---

## 🚀 Production Deployment

### Requirements

```toml
python = "^3.10"
torch = "2.9.1"
numpy = "2.4.0"
pandas = "2.3.3"
duckdb = "1.4.3"
matplotlib = "*"
```

### Key Configuration

```python
# src/common/config.py
DB_PATH = Path("data/01_raw/savant.duckdb")
DECISION_TIME = 0.167  # seconds
```

### Error Handling

- ✅ Missing pitcher data → ValueError
- ✅ Unknown pitch type → ValueError + Fallback
- ✅ DuckDB connection → Context manager auto-close
- ✅ Empty DataFrame → Early exit with warning

---

## 📚 References

### Scientific Background

1. **Alan Nathan Model**: Baseball aerodynamics with spin saturation
2. **Decision Point**: 167ms based on human reaction time studies
3. **Tunnel Score**: Modified sigmoid function for similarity

### MLB Applications

- Pitch sequencing optimization
- Batter preparation scouting
- Pitcher development feedback

---

## 🎓 Future Enhancements

### Potential Improvements

1. **PINN Integration**: Use trained neural network for faster simulation
2. **Multi-Pitch Sequences**: 3+ pitch optimization
3. **Batter-Specific Models**: Adjust Decision Point by batter skill
4. **GPU Acceleration**: Batch process 1000+ pitches simultaneously

### API Extension

```python
# Future API design
analyzer.optimize_sequence(
    pitcher_id=621111,
    sequence_length=3,
    batter_profile=elite_hitter
)
# Output: ['FF', 'SL', 'CH'] with scores
```

---

## ✅ Verification Checklist

- [x] Data integration with AegisDataLoader
- [x] get_pitch_profile extracts real DNA
- [x] Delta Injection method implemented
- [x] calculate_approach_angles (VAA, HAA)
- [x] Visualization shows VAA/HAA
- [x] Real data testing (Pitcher 621111)
- [x] Error handling complete
- [x] Documentation comprehensive
- [x] Production ready

---

## 📝 Change Log

### Version 1.0.0 (2026-01-06)

- ✅ Refactored from prototype to production
- ✅ Added get_pitch_profile method
- ✅ Implemented Delta Injection algorithm
- ✅ Added VAA/HAA calculation
- ✅ Enhanced visualization with approach angles
- ✅ Improved error handling
- ✅ Comprehensive documentation

---

**Status**: ✅ **PRODUCTION READY**

**Maintainer**: Chief Engineer
**Contact**: user@yonsei.ac.kr
**Repository**: aegis-pitching-engine
