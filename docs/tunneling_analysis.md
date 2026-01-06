# Tunneling Analysis 구현 완료

## 📅 구현 날짜

2026년 1월 6일

## 🎯 개요

**Tunneling(터널링)**은 두 개 이상의 다른 투구가 초기 궤적은 동일하지만 나중에 다른 경로로 분기되는 현상입니다. 타자는 초기에 투구를 구분할 수 없어 의사결정이 어려워집니다.

### 핵심 개념

1. **Decision Point**: 투구 후 0.167초 (약 23.8ft, 7.25m)

   - 타자가 스윙 여부를 결정하는 임계 시점
   - 이 시점 이후로는 궤적 변화에 반응하기 어려움

2. **Tunnel Score**: 두 투구의 유사도 측정

   ```
   Score = 1 / (1 + Distance)
   ```

   - Distance: Decision Point에서의 3D 유클리드 거리
   - Score가 1에 가까울수록 터널링 효과가 큼

3. **Counterfactual Simulation**: 반사실적 시뮬레이션
   - 신체적 조건(릴리즈 포인트, 팔 각도)은 동일
   - 투구 타입(속도, 회전)만 변경하여 시뮬레이션

## 🔬 구현 내용

### 1. TunnelingAnalyzer 클래스

**파일**: [src/game_theory/tunneling.py](../src/game_theory/tunneling.py)

#### 주요 메서드

##### `simulate_counterfactual(actual_pitch_data, target_pitch_type)`

실제 투구 데이터를 반사실적 투구로 변환

**신체적 조건 유지 (Kinematics):**

- `release_pos_x, release_pos_y, release_pos_z`: 릴리즈 위치
- `release_extension`: 팔 길이
- 투구 메커니즘 (arm slot, release angle)

**변경 사항 (Pitch Characteristics):**

- `release_speed`: 투구 타입별 평균 속도
- `spin_rate`: 회전 속도
- `spin_axis`: 회전 축

**투구 타입 프로필:**

```python
PITCH_TYPE_PROFILES = {
    'FF': {  # 4-Seam Fastball
        'spin_rate': 2300 RPM,
        'velocity_modifier': 1.0,
        'spin_axis': (1.0, 0, 0)  # Backspin
    },
    'SL': {  # Slider
        'spin_rate': 2500 RPM,
        'velocity_modifier': 0.90,
        'spin_axis': (0.5, 0, 0.866)  # Gyro + Sidespin
    },
    ...
}
```

##### `calculate_tunnel_score(traj1, time1, traj2, time2)`

터널링 점수 계산

**프로세스:**

1. Decision Point (0.167s)에서 각 궤적의 위치 보간
2. 3D 유클리드 거리 계산
   ```
   Distance = √[(x₁-x₂)² + (y₁-y₂)² + (z₁-z₂)²]
   ```
3. 터널 점수 계산
   ```
   Score = 1 / (1 + Distance)
   ```

**반환 값:**

```python
{
    'tunnel_score': 0.844,           # 0~1
    'distance_at_decision': 0.184,    # meters
    'decision_point_pos1': [x, y, z],
    'decision_point_pos2': [x, y, z]
}
```

##### `visualize_tunneling(result, save_path)`

타자 시점 시각화

**2개의 그래프:**

1. **Side View (Y-Z)**: 측면에서 본 궤적
2. **Batter's View (X-Z)**: 타자가 보는 정면 뷰

**표시 요소:**

- 실제 궤적 (파란색 실선)
- 반사실적 궤적 (빨간색 점선)
- Decision Point (마커)
- 스트라이크 존 (녹색)
- 터널 점수 (제목)

### 2. 물리 기반 궤적 시뮬레이션

**Euler 적분법:**

```python
while t < max_time:
    forces = physics_engine.compute_forces(state, spin)
    accel = forces / mass

    velocity += accel * dt
    position += velocity * dt

    t += dt
```

**힘 계산:**

- 중력: F_g = -mg
- 항력: F_d = -½ρAC_D|v|v
- 마그누스: F_m = ½ρAC_L(ω×v)

## 📊 실험 결과

### 테스트 조건

- **실제 투구**: Fastball, 97.2 mph, 2419 RPM
- **비교 대상**: SI, FC, SL, CU, CH
- **Decision Point**: 0.167s

### 터널링 점수

| 조합        | Tunnel Score | Distance (m) | 해석           |
| ----------- | ------------ | ------------ | -------------- |
| **FF → SI** | **0.844**    | **0.184**    | 🏆 최고 터널링 |
| FF → FC     | 0.728        | 0.374        | 좋은 터널링    |
| FF → SL     | 0.528        | 0.896        | 보통           |
| FF → CH     | 0.475        | 1.107        | 약간 구분됨    |
| FF → CU     | 0.395        | 1.534        | 명확히 구분됨  |

### 분석

1. **FF ↔ SI (Sinker)**

   - Score: 0.844 (가장 높음)
   - 이유: 속도 차이가 작고 (98% vs 100%), 초기 궤적이 거의 동일
   - 실전 의미: 타자가 구분하기 매우 어려움

2. **FF ↔ FC (Cutter)**

   - Score: 0.728
   - 이유: 속도는 유사하지만 약간의 측면 움직임 차이
   - 실전 의미: 효과적인 조합

3. **FF ↔ SL (Slider)**

   - Score: 0.528
   - 이유: 속도 차이 10%, 회전 축 차이로 궤적 분기
   - 실전 의미: 전통적인 터널링 조합

4. **FF ↔ CU (Curveball)**
   - Score: 0.395
   - 이유: 속도 차이 17%, Topspin vs Backspin
   - 실전 의미: 타자가 구분 가능하지만 여전히 효과적

## 🎨 시각화 결과

### 생성된 그래프

[examples/tunneling_analysis.png](../examples/tunneling_analysis.png)

**특징:**

- 좌측: 측면 뷰 (Y-Z 평면)
- 우측: 타자 시점 (X-Z 평면)
- Decision Point에서의 위치 차이 명확히 표시
- 스트라이크 존 오버레이

## 🚀 사용 방법

### 기본 사용

```python
from src.game_theory import TunnelingAnalyzer
from src.data_pipeline import AegisDataLoader

# 데이터 로드
with AegisDataLoader() as loader:
    df = loader.load_data_by_year(year=2024, limit=100)
    fastball_data = df[df['pitch_type'] == 'FF'].iloc[0]

# Analyzer 초기화
analyzer = TunnelingAnalyzer()

# 반사실적 시뮬레이션
result = analyzer.simulate_counterfactual(
    actual_pitch_data=fastball_data,
    target_pitch_type='SL'
)

# 터널 점수 계산
tunnel_info = analyzer.calculate_tunnel_score(
    result['actual_traj'], result['actual_time'],
    result['cf_traj'], result['cf_time']
)

print(f"Tunnel Score: {tunnel_info['tunnel_score']:.3f}")

# 시각화
analyzer.visualize_tunneling(result, save_path='output.png')
```

### 여러 조합 비교

```python
target_types = ['SI', 'FC', 'SL', 'CU', 'CH']

for target in target_types:
    result = analyzer.simulate_counterfactual(fastball_data, target)
    tunnel_info = analyzer.calculate_tunnel_score(
        result['actual_traj'], result['actual_time'],
        result['cf_traj'], result['cf_time']
    )
    print(f"FF → {target}: {tunnel_info['tunnel_score']:.3f}")
```

## 📐 수학적 정의

### Decision Point 위치 보간

주어진 시간 t에서 위치 계산 (선형 보간):

```
P(t) = P₀ + (t - t₀)/(t₁ - t₀) × (P₁ - P₀)
```

### 유클리드 거리

```
D = √[(x₁-x₂)² + (y₁-y₂)² + (z₁-z₂)²]
```

### 터널 점수

```
S = 1/(1 + D)
```

**특성:**

- D = 0 → S = 1.0 (완벽한 터널링)
- D = 1 → S = 0.5
- D → ∞ → S → 0

## 🎓 이론적 배경

### 타자의 의사결정 모델

**시간 제약:**

- 95mph Fastball: 홈플레이트까지 약 0.4초
- Decision Point: 0.167초 (약 40%)
- 반응 시간 필요: 약 0.15초
- 실제 결정 가능 시간: 매우 짧음

**정보 처리:**

1. **초기 단계** (0~0.167s): 궤적 예측
2. **Decision Point**: 스윙 여부 결정
3. **후기 단계**: 궤적 조정 (거의 불가능)

### 터널링의 심리학

1. **인지 부하**: 두 투구가 비슷할수록 구분 어려움
2. **예측 오류**: 초기 궤적 기반 예측이 틀림
3. **스윙 결정 지연**: 확신이 없어 늦은 결정

## 🔬 검증 및 한계

### 검증

- ✅ 물리 기반 시뮬레이션 (검증된 Alan Nathan Model)
- ✅ Decision Point 시간 (생리학적 근거)
- ✅ 터널 점수와 실제 효과성 상관관계 (문헌 기반)

### 한계 및 개선 방향

1. **Spin Axis 단순화**

   - 현재: 투구 타입별 평균 회전 축
   - 개선: 실제 spin axis 데이터 사용

2. **타자 특성 미반영**

   - 현재: 일반적인 Decision Point
   - 개선: 타자별 반응 시간 차이 고려

3. **환경 요소**

   - 현재: 표준 환경만 고려
   - 개선: 구장 고도, 날씨 효과

4. **Biomechanical 제약**
   - 현재: 모든 투구 조합 가능 가정
   - 개선: 투수별 가능한 조합만 분석

## 📚 참고 문헌

1. Nathan, A. M. (2008). "The effect of spin on the flight of a baseball." _American Journal of Physics_.

2. Gray, R. (2002). "Behavior of college baseball players in a virtual batting task." _Journal of Experimental Psychology: Human Perception and Performance_.

3. Bahill, A. T., & LaRitz, T. (1984). "Why can't batters keep their eyes on the ball?" _American Scientist_.

4. Walsh, M. (2017). "Pitch Tunneling and Why the Traditional Strike Zone Doesn't Matter." _The Hardball Times_.

## ✅ 체크리스트

- [x] TunnelingAnalyzer 클래스 구현
- [x] simulate_counterfactual() 메서드
- [x] calculate_tunnel_score() 메서드
- [x] Decision Point (0.167s) 적용
- [x] 유클리드 거리 계산
- [x] 터널 점수 공식 구현
- [x] 타자 시점 시각화 (Side View + Batter's View)
- [x] Decision Point 마커 표시
- [x] 스트라이크 존 오버레이
- [x] 여러 투구 타입 비교
- [x] 실제 Statcast 데이터 테스트
- [x] Type hints 및 Docstring

---

**작성자**: Aegis Game Theory Team
**버전**: 1.0
**최종 수정**: 2026년 1월 6일
