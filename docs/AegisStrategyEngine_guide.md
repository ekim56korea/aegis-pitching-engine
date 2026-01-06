# AegisStrategyEngine 사용 가이드

## 개요

**AegisStrategyEngine**은 물리 엔진, 게임 이론, 분석 모듈을 통합한 Config-Driven 투구 전략 의사결정 엔진입니다.

### 핵심 특징

✅ **Config-Driven**: 모든 파라미터는 `StrategyConfig`에서 로드 (하드코딩 금지)
✅ **Dynamic Action Space**: 투수 레퍼토리에 맞춘 가능한 행동만 생성
✅ **Multi-Metric Simulation**: Tunneling, EV, Command, Stuff 종합 평가
✅ **Probabilistic Selection**: Softmax 기반 확률적 선택 (Temperature 조절)
✅ **Rationale Generation**: 의사결정 이유를 자연어로 설명

---

## 지원 구종 (MLB 전체)

```python
MLB_PITCH_TYPES = {
    'FF': 'Four-Seam Fastball',
    'SI': 'Sinker',
    'FC': 'Cutter',
    'SL': 'Slider',
    'ST': 'Sweeper',
    'CU': 'Curveball',
    'KC': 'Knuckle Curve',
    'CH': 'Changeup',
    'FS': 'Splitter',
    'FO': 'Forkball',
    'KN': 'Knuckleball'
}
```

---

## Zone 시스템 (13 Zones)

### Zone 분류

| Zone Type  | Description                        | Risk Level |
| ---------- | ---------------------------------- | ---------- |
| **Heart**  | Strike Zone 중앙 (타격 확률 높음)  | High       |
| **Shadow** | Strike Zone 가장자리 (유효한 공략) | Medium     |
| **Chase**  | 볼존이지만 유인 가능               | Low        |
| **Waste**  | 명백한 볼 (카운트 악화용)          | Very Low   |

### Zone 상세

```
Heart Zones (3):
- heart_high, heart_mid, heart_low

Shadow Zones (6):
- shadow_in_high, shadow_in_mid, shadow_in_low
- shadow_out_high, shadow_out_mid, shadow_out_low

Chase Zones (4):
- chase_high, chase_low, chase_in, chase_out

Waste Zone (1):
- waste (특수 상황에서만 사용)
```

---

## 기본 사용법

### 1. 엔진 초기화

```python
from src.game_theory import AegisStrategyEngine
from src.common.config import StrategyConfig

# 기본 설정으로 초기화
engine = AegisStrategyEngine(device='cpu')

# 또는 커스텀 설정 사용
custom_config = StrategyConfig()
custom_config.FEATURE_WEIGHTS['tunneling'] = 0.4  # 터널링 가중치 증가
engine = AegisStrategyEngine(config=custom_config, device='cpu')
```

### 2. 투구 결정 요청

```python
# 게임 상태 정의
game_state = {
    'outs': 2,              # 아웃 카운트 (0-2)
    'count': '3-2',         # 볼-스트라이크 카운트
    'runners': [1, 1, 1],   # 주자 상태 [1루, 2루, 3루]
    'score_diff': -1,       # 점수 차이 (음수 = 지고 있음)
    'inning': 9             # 이닝 (1-9+)
}

# 투수 상태 정의
pitcher_state = {
    'hand': 'R',            # 'R' (우투) or 'L' (좌투)
    'role': 'RP',           # 'SP' (선발) or 'RP' (불펜)
    'pitch_count': 28,      # 현재 투구수
    'entropy': 0.65,        # 엔트로피 (0-1, 높을수록 예측 불가)
    'prev_pitch': 'SL',     # 직전 투구 구종
    'prev_velo': 85.0       # 직전 투구 구속 (mph)
}

# 매치업 정보 정의
matchup_state = {
    'batter_hand': 'L',     # 'L' (좌타) or 'R' (우타)
    'times_faced': 1,       # TTO (0-3+)
    'chase_rate': 0.38,     # O-Swing% (0-1)
    'whiff_rate': 0.32,     # Whiff% (0-1)
    'iso': 0.220,           # Isolated Power
    'gb_fb_ratio': 0.9,     # Ground Ball / Fly Ball Ratio
    'ops': 0.810            # On-base Plus Slugging
}

# 투수 레퍼토리 (실제로 던질 수 있는 구종만)
available_pitches = ['FF', 'SL', 'CH']

# 투수 통계 (선택사항, 없으면 리그 평균 사용)
pitcher_stats = {
    'stuff_plus': {
        'FF': 105.0,        # Four-Seam Stuff+ (100 = 평균)
        'SL': 115.0,        # Slider Stuff+ (뛰어남)
        'CH': 98.0          # Changeup Stuff+
    },
    'zone_command': {       # 존별 제구 성공률 (0-1)
        'FF': {'chase_low': 0.70, 'shadow_out_mid': 0.75},
        'SL': {'chase_low': 0.68, 'chase_out': 0.72},
        'CH': {'chase_low': 0.65}
    }
}

# 의사결정 요청
result = engine.decide_pitch(
    game_state,
    pitcher_state,
    matchup_state,
    available_pitches,
    pitcher_stats  # 선택사항
)
```

### 3. 결과 해석

```python
# 선택된 행동
print(f"구종: {result.selected_action.pitch_type}")
print(f"존: {result.selected_action.zone}")
print(f"좌표: ({result.selected_action.plate_x:.2f}, {result.selected_action.plate_z:.2f})")

# Leverage 수준
print(f"상황: {result.leverage_level}")
# 'high_leverage', 'medium_leverage', 'low_leverage'

# 엔트로피 상태
print(f"패턴 노출도: {result.entropy_status}")
# 'low', 'medium', 'high'

# 상위 행동들의 확률 분포
print("\nTop 5 Actions:")
for action_key, prob in result.action_probs.items():
    print(f"  {action_key}: {prob:.1%}")

# Q-value 분포
print("\nTop 5 Q-Values:")
for action_key, q_val in result.q_values.items():
    print(f"  {action_key}: {q_val:.3f}")

# 자연어 설명
print(f"\n의사결정 이유:\n{result.rationale}")
```

---

## 고급 사용법

### 1. Feature Weights 커스터마이징

```python
from src.common.config import StrategyConfig

config = StrategyConfig()

# 가중치 조정 (합이 1.0일 필요는 없음)
config.FEATURE_WEIGHTS = {
    'tunneling': 0.35,      # 터널링 중시
    'ev_delta': 0.25,       # EV 차이 중시
    'chase_rate': 0.15,     # Chase 유도
    'stuff_quality': 0.15,  # Stuff+ 중시
    'command': 0.05,        # 제구 낮게 평가
    'entropy': 0.05         # 엔트로피 낮게 평가
}

engine = AegisStrategyEngine(config=config)
```

### 2. Temperature 조정

```python
config = StrategyConfig()

# Softmax Temperature 조정
config.TEMPERATURE_CONFIG = {
    'high_leverage': 0.2,    # 더 확실한 선택 (낮은 τ)
    'medium_leverage': 0.5,
    'low_leverage': 1.0,     # 더 많은 탐색 (높은 τ)
    'exploration': 1.5       # 극단적 탐색
}

engine = AegisStrategyEngine(config=config)
```

### 3. Exploitation Multiplier 조정

```python
config = StrategyConfig()

# 타자 약점 공략 시 가중치 증폭 배율
config.EXPLOITATION_CONFIG = {
    'hot_zone_multiplier': 2.0,      # Hot Zone 공략 강화
    'weak_zone_multiplier': 2.5,     # Weak Zone 공략 강화
    'high_whiff_multiplier': 2.0,    # 헛스윙률 높은 구종 강화
    'low_contact_multiplier': 1.8    # 컨택률 낮은 존 강화
}

engine = AegisStrategyEngine(config=config)
```

---

## Decision Logic 상세

### Step 1: Context Awareness

1. **상태 벡터화**: `ContextEncoder`로 현재 상태를 42-dim 텐서로 변환
2. **Leverage 판단**: 점수 차이, 이닝, 주자 상황 등으로 high/medium/low 결정
3. **엔트로피 확인**: 패턴 노출도 평가 (낮으면 패턴 변경 필요)

### Step 2: Action Space Generation

- 투수 레퍼토리에 있는 구종만 사용
- 각 구종 × 13개 존 = 최대 55개 행동 (구종당)
- Waste zone은 특수 상황에서만 포함

### Step 3: Simulation (Metric Calculation)

각 행동에 대해 다음 메트릭 계산:

| Metric              | Description               | Range  |
| ------------------- | ------------------------- | ------ |
| **Tunneling Score** | 직전 투구와의 궤적 유사성 | [0, 1] |
| **EV Delta**        | Effective Velocity 차이   | mph    |
| **Command Risk**    | 제구 성공률               | [0, 1] |
| **Stuff Quality**   | 구종 위력 (Stuff+)        | 70-130 |
| **Chase Score**     | 헛스윙 유도 점수          | [0, 1] |
| **Entropy Bonus**   | 패턴 변경 보너스          | [0, 1] |

### Step 4: Payoff Calculation (Q-Value)

```
Q(s,a) = Σ(w_i · feature_i) × exploitation_multiplier

where:
  w_i: Feature weight (from Config)
  feature_i: Normalized metric value
  exploitation_multiplier: Batter weakness multiplier
```

### Step 5: Probabilistic Selection (Softmax)

```
P(a) = exp(Q(s,a) / τ) / Σ exp(Q(s,a') / τ)

where:
  τ (temperature): Leverage에 따라 조절
    - High Leverage: 낮은 τ → 확실한 선택
    - Low Leverage: 높은 τ → 다양한 탐색
```

### Step 6: Rationale Generation

자연어로 의사결정 이유 설명:

- 직전 투구 정보
- 터널링 점수
- EV 차이
- 타자 약점 (Chase Rate, Whiff Rate)
- 선택된 구종 및 존
- Leverage 상황
- 대안 행동들

---

## Config 파라미터 전체 목록

### Feature Weights

```python
FEATURE_WEIGHTS = {
    'tunneling': 0.30,
    'ev_delta': 0.20,
    'chase_rate': 0.15,
    'stuff_quality': 0.20,
    'command': 0.10,
    'entropy': 0.05
}
```

### Exploitation Config

```python
EXPLOITATION_CONFIG = {
    'hot_zone_multiplier': 1.5,
    'weak_zone_multiplier': 2.0,
    'high_whiff_multiplier': 1.8,
    'low_contact_multiplier': 1.6
}
```

### Temperature Config

```python
TEMPERATURE_CONFIG = {
    'high_leverage': 0.3,
    'medium_leverage': 0.5,
    'low_leverage': 0.8,
    'exploration': 1.2
}
```

### Leverage Thresholds

```python
LEVERAGE_THRESHOLDS = {
    'high_leverage_score_diff': 2,
    'high_leverage_inning': 7,
    'high_leverage_runners': 2,
    'critical_count': ['3-2', '3-1', '2-2']
}
```

### Entropy Thresholds

```python
ENTROPY_THRESHOLDS = {
    'low_entropy': 0.5,
    'medium_entropy': 0.7,
    'high_entropy': 0.7
}
```

### Command Config

```python
COMMAND_CONFIG = {
    'league_average_command': 0.65,
    'excellent_command': 0.80,
    'poor_command': 0.50,
    'command_penalty': 0.5
}
```

---

## 실전 예시

### 예시 1: 승부처 (9회 만루 2아웃)

```python
engine = AegisStrategyEngine()

game_state = {
    'outs': 2,
    'count': '3-2',
    'runners': [1, 1, 1],
    'score_diff': -1,
    'inning': 9
}

pitcher_state = {
    'hand': 'R',
    'role': 'RP',
    'pitch_count': 28,
    'entropy': 0.65,
    'prev_pitch': 'SL',
    'prev_velo': 85.0
}

matchup_state = {
    'batter_hand': 'L',
    'times_faced': 1,
    'chase_rate': 0.38,
    'whiff_rate': 0.32,
    'iso': 0.220,
    'gb_fb_ratio': 0.9,
    'ops': 0.810
}

result = engine.decide_pitch(
    game_state, pitcher_state, matchup_state,
    available_pitches=['FF', 'SL', 'CH']
)

# 출력 예시:
# Selected: FF @ chase_out
# Rationale: "직전 Slider(SL) 이후, EV 차이가 +4.2mph로 크며,
#            타자의 Chase Rate이 38.0%로 높아, 헛스윙률이 32.0%로 높아,
#            Four-Seam Fastball(FF)를 chase_out 존에 선택함,
#            현재 승부처 상황으로 확실한 공을 선택했습니다."
```

### 예시 2: 여유 상황 (초반, 큰 점수 차)

```python
game_state = {
    'outs': 0,
    'count': '1-1',
    'runners': [0, 0, 0],
    'score_diff': 5,
    'inning': 3
}

pitcher_state = {
    'hand': 'L',
    'role': 'SP',
    'pitch_count': 45,
    'entropy': 0.88,
    'prev_pitch': 'FF',
    'prev_velo': 92.0
}

# Temperature가 높아져 더 다양한 선택 시도
result = engine.decide_pitch(
    game_state, pitcher_state, matchup_state,
    available_pitches=['FF', 'SI', 'SL', 'CH']
)

# 출력 예시:
# Selected: CH @ heart_low
# Rationale: "직전 Four-Seam Fastball(FF) 이후, 터널링 점수가 0.85로 높고,
#            EV 차이가 +7.0mph로 크며, Changeup(CH)를 heart_low 존에 선택함,
#            여유 있는 상황으로 다양한 선택을 시도했습니다."
```

---

## 제약 사항

1. **투수 레퍼토리 엄수**: `available_pitches`에 없는 구종은 절대 추천하지 않음
2. **최소 레퍼토리**: 최소 3개 구종 보유 권장 (MIN_PITCH_REPERTOIRE_SIZE)
3. **통계 데이터**: `pitcher_stats`가 없으면 리그 평균 사용 (성능 저하 가능)
4. **Waste Zone**: 현재 기본적으로 제외 (특수 로직 필요 시 추가)

---

## 트러블슈팅

### 문제: "No valid actions generated"

**원인**: `available_pitches`가 비어있거나, Config에 정의되지 않은 구종 코드 사용

**해결책**:

```python
# 올바른 구종 코드 사용
available_pitches = ['FF', 'SL', 'CH']  # ✅

# 잘못된 코드
available_pitches = ['4FB', 'SLD', 'CHG']  # ❌
```

### 문제: 항상 같은 구종만 선택

**원인**: Temperature가 너무 낮거나, 특정 Feature Weight가 과도하게 높음

**해결책**:

```python
config = StrategyConfig()
config.TEMPERATURE_CONFIG['high_leverage'] = 0.5  # 증가
config.FEATURE_WEIGHTS['tunneling'] = 0.25  # 감소
```

### 문제: 이상한 존 선택 (Heart Zone 과다)

**원인**: Command Risk 가중치가 너무 높거나, Zone Risk가 반영되지 않음

**해결책**: Exploitation Multiplier와 Command Config 재조정

---

## API Reference

### AegisStrategyEngine

```python
class AegisStrategyEngine:
    def __init__(self, config: Optional[StrategyConfig] = None, device: str = 'cpu')

    def decide_pitch(
        self,
        game_state: Dict,
        pitcher_state: Dict,
        matchup_state: Dict,
        available_pitches: List[str],
        pitcher_stats: Optional[Dict] = None
    ) -> DecisionResult
```

### DecisionResult

```python
@dataclass
class DecisionResult:
    selected_action: Action
    action_probs: Dict[str, float]
    q_values: Dict[str, float]
    rationale: str
    leverage_level: str
    entropy_status: str
```

### Action

```python
@dataclass
class Action:
    pitch_type: str
    zone: str
    plate_x: float
    plate_z: float

    def to_dict(self) -> Dict
```

---

## 성능 최적화

### CPU vs GPU

```python
# CPU (기본)
engine = AegisStrategyEngine(device='cpu')

# GPU (대규모 시뮬레이션용)
engine = AegisStrategyEngine(device='cuda')
```

### 배치 처리

현재는 단일 결정만 지원하지만, 향후 배치 처리 추가 예정:

```python
# 향후 지원 예정
results = engine.decide_pitch_batch(
    game_states_list,
    pitcher_states_list,
    matchup_states_list,
    available_pitches_list
)
```

---

## 라이선스 및 기여

**Status**: Internal Development Tool
**Author**: Aegis Development Team
**Contact**: user@yonsei.ac.kr

---

**마지막 업데이트**: 2026-01-06
