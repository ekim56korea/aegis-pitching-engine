"""
Aegis Pitching Engine 설정 파일
프로젝트 전체에서 사용되는 경로, 상수, 설정값 관리
"""

from pathlib import Path

# 프로젝트 루트 디렉토리
PROJECT_ROOT = Path(__file__).parent.parent.parent

# 데이터 경로
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "01_raw"
INTERMEDIATE_DATA_DIR = DATA_DIR / "02_intermediate"
PRIMARY_DATA_DIR = DATA_DIR / "03_primary"
FEATURE_STORE_DIR = DATA_DIR / "04_feature_store"

# DuckDB 경로
DB_PATH = RAW_DATA_DIR / "savant.duckdb"

# 필수 컬럼 정의 (Baseball Savant 데이터 기준)
REQUIRED_COLUMNS = [
    "pitcher",
    "batter",
    "pitch_type",
    "release_speed",
    "release_pos_x",
    "release_pos_y",
    "release_pos_z",
    "pfx_x",
    "pfx_z",
    "plate_x",
    "plate_z",
    "vx0",
    "vy0",
    "vz0",
    "ax",
    "ay",
    "az",
    "sz_top",
    "sz_bot",
    "stand",
    "p_throws",
    "game_year",
    "release_spin_rate",
    "release_extension",
]

# 물리 엔진 관련 상수
GRAVITY = 9.80665  # m/s^2 (중력 가속도)
AIR_DENSITY = 1.225  # kg/m^3 (공기 밀도, 해수면 기준)
BASEBALL_MASS = 0.145  # kg (야구공 질량)
BASEBALL_DIAMETER = 0.074  # m (야구공 지름)

# 모델 설정
MODEL_CONFIG_DIR = PROJECT_ROOT / "config" / "model"
PROCESS_CONFIG_DIR = PROJECT_ROOT / "config" / "process"

# ============================================================================
# Strategy Engine Configuration (Config-Driven, No Hardcoding)
# ============================================================================

class StrategyConfig:
    """
    AegisStrategyEngine의 모든 파라미터를 정의하는 설정 클래스
    하드코딩 방지를 위해 모든 가중치, 임계값, 상수를 중앙 관리
    """

    # ========================================================================
    # MLB Pitch Types (전체 구종)
    # ========================================================================
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

    # ========================================================================
    # Zone Definitions (MLB Savant 기준 13-Zone System)
    # ========================================================================
    ZONES = {
        # Heart Zones (Strike Zone 중앙, 높은 타격 확률)
        'heart_high': {'x_range': (-0.33, 0.33), 'z_range': (2.5, 3.5), 'risk': 'high'},
        'heart_mid': {'x_range': (-0.33, 0.33), 'z_range': (1.5, 2.5), 'risk': 'high'},
        'heart_low': {'x_range': (-0.33, 0.33), 'z_range': (1.5, 2.5), 'risk': 'medium'},

        # Shadow Zones (Strike Zone 가장자리, 유효한 공략)
        'shadow_in_high': {'x_range': (-0.83, -0.33), 'z_range': (2.5, 3.5), 'risk': 'medium'},
        'shadow_in_mid': {'x_range': (-0.83, -0.33), 'z_range': (1.5, 2.5), 'risk': 'medium'},
        'shadow_in_low': {'x_range': (-0.83, -0.33), 'z_range': (1.5, 2.5), 'risk': 'low'},
        'shadow_out_high': {'x_range': (0.33, 0.83), 'z_range': (2.5, 3.5), 'risk': 'medium'},
        'shadow_out_mid': {'x_range': (0.33, 0.83), 'z_range': (1.5, 2.5), 'risk': 'medium'},
        'shadow_out_low': {'x_range': (0.33, 0.83), 'z_range': (1.5, 2.5), 'risk': 'low'},

        # Chase Zones (볼존이지만 유인 가능)
        'chase_high': {'x_range': (-0.83, 0.83), 'z_range': (3.5, 4.0), 'risk': 'low'},
        'chase_low': {'x_range': (-0.83, 0.83), 'z_range': (1.0, 1.5), 'risk': 'low'},
        'chase_in': {'x_range': (-1.5, -0.83), 'z_range': (1.5, 3.5), 'risk': 'low'},
        'chase_out': {'x_range': (0.83, 1.5), 'z_range': (1.5, 3.5), 'risk': 'low'},

        # Waste Zones (명백한 볼, 카운트 악화용)
        'waste': {'x_range': (-2.0, 2.0), 'z_range': (0.5, 4.5), 'risk': 'very_low'}
    }

    # ========================================================================
    # Payoff Calculation Weights (Q-Value Feature Weights)
    # ========================================================================
    FEATURE_WEIGHTS = {
        'tunneling': 0.30,      # 터널링 점수 가중치
        'ev_delta': 0.20,       # Effective Velocity 차이 가중치
        'chase_rate': 0.15,     # 타자 Chase Rate 가중치
        'stuff_quality': 0.20,  # 구종 위력(Stuff+) 가중치
        'command': 0.10,        # 제구력(Command) 가중치
        'entropy': 0.05         # 엔트로피(패턴 변경) 가중치
    }

    # ========================================================================
    # Exploitation Multiplier (타자 약점 공략 시 가중치 증폭)
    # ========================================================================
    EXPLOITATION_CONFIG = {
        'hot_zone_multiplier': 1.5,      # 타자 Hot Zone 공략 시 배율
        'weak_zone_multiplier': 2.0,     # 타자 Weak Zone 공략 시 배율
        'high_whiff_multiplier': 1.8,    # 헛스윙률 높은 구종 선택 시 배율
        'low_contact_multiplier': 1.6    # 컨택률 낮은 존 공략 시 배율
    }

    # ========================================================================
    # Temperature (Softmax Selection)
    # ========================================================================
    TEMPERATURE_CONFIG = {
        'high_leverage': 0.3,    # 승부처: 낮은 τ (확실한 선택)
        'medium_leverage': 0.5,  # 중간 상황: 중간 τ
        'low_leverage': 0.8,     # 여유 상황: 높은 τ (탐색)
        'exploration': 1.2       # 탐색 모드: 매우 높은 τ
    }

    # ========================================================================
    # Leverage Thresholds (상황 판단 임계값)
    # ========================================================================
    LEVERAGE_THRESHOLDS = {
        'high_leverage_score_diff': 2,   # 점수 차이 ≤ 2점이면 high leverage
        'high_leverage_inning': 7,       # 7회 이상이면 high leverage
        'high_leverage_runners': 2,      # 주자 2명 이상이면 high leverage
        'critical_count': ['3-2', '3-1', '2-2']  # 중요한 카운트
    }

    # ========================================================================
    # Entropy Thresholds (패턴 노출도 판단)
    # ========================================================================
    ENTROPY_THRESHOLDS = {
        'low_entropy': 0.5,      # 엔트로피 < 0.5 → 패턴 노출 위험
        'medium_entropy': 0.7,   # 0.5 ≤ 엔트로피 < 0.7 → 주의
        'high_entropy': 0.7      # 엔트로피 ≥ 0.7 → 예측 불가
    }

    # ========================================================================
    # Command Risk (제구 성공률 기준값)
    # ========================================================================
    COMMAND_CONFIG = {
        'league_average_command': 0.65,  # 리그 평균 제구 성공률 (Fallback)
        'excellent_command': 0.80,       # 우수한 제구
        'poor_command': 0.50,            # 낮은 제구
        'command_penalty': 0.5           # 제구 실패 시 패널티 배율
    }

    # ========================================================================
    # Stuff+ Baseline (구종별 위력 기준값)
    # ========================================================================
    STUFF_BASELINE = {
        'FF': 100.0,  # Four-Seam Fastball
        'SI': 100.0,  # Sinker
        'FC': 100.0,  # Cutter
        'SL': 100.0,  # Slider
        'ST': 100.0,  # Sweeper
        'CU': 100.0,  # Curveball
        'KC': 100.0,  # Knuckle Curve
        'CH': 100.0,  # Changeup
        'FS': 100.0,  # Splitter
        'FO': 100.0,  # Forkball
        'KN': 100.0   # Knuckleball
    }

    # ========================================================================
    # Action Space Limits (행동 공간 제약)
    # ========================================================================
    MAX_ACTIONS_PER_DECISION = 55  # 최대 행동 수 (11 pitches × 5 zones)
    MIN_PITCH_REPERTOIRE_SIZE = 3   # 최소 구종 보유 수

    # ========================================================================
    # Data Noise Filtering (Robust to Trackman Misclassification)
    # ========================================================================
    MIN_PITCH_USAGE_THRESHOLD = 0.03  # 최소 구사율 (3% 미만은 Noise로 간주)
    MIN_SAMPLE_SIZE_THRESHOLD = 10     # Stuff+ 계산을 위한 최소 샘플 수
    LOW_SAMPLE_PENALTY = 0.7           # 샘플 부족 시 Stuff+ 페널티 배율
    NOISE_LOGGING_ENABLED = True       # Noise 필터링 로그 활성화

    # ========================================================================
    # Rationale Generation (자연어 설명 생성 설정)
    # ========================================================================
    RATIONALE_CONFIG = {
        'tunneling_threshold': 0.85,     # 터널링 점수 높음 판단 기준
        'ev_significant_delta': 3.0,     # EV 차이 유의미 판단 (mph)
        'chase_high_threshold': 0.35,    # Chase Rate 높음 판단 (35%)
        'stuff_excellent_threshold': 110, # Stuff+ 우수 판단 기준
        'top_k_alternatives': 5          # 상위 K개 대안 표시
    }

    # ========================================================================
    # Normalization Constants (정규화 상수)
    # ========================================================================
    NORMALIZATION = {
        'tunneling_max': 1.0,            # 터널링 점수 최대값
        'ev_delta_range': 10.0,          # EV 차이 정규화 범위 (±10mph)
        'stuff_plus_mean': 100.0,        # Stuff+ 평균
        'stuff_plus_std': 15.0,          # Stuff+ 표준편차
        'chase_rate_max': 1.0,           # Chase Rate 최대값
        'command_max': 1.0               # 제구 성공률 최대값
    }

    @classmethod
    def get_feature_weight(cls, feature_name: str) -> float:
        """특정 feature의 가중치 반환"""
        return cls.FEATURE_WEIGHTS.get(feature_name, 0.0)

    @classmethod
    def get_temperature(cls, leverage_level: str) -> float:
        """Leverage 수준에 따른 Temperature 반환"""
        return cls.TEMPERATURE_CONFIG.get(leverage_level, 0.5)

    @classmethod
    def is_high_leverage(cls, game_state: dict) -> bool:
        """High Leverage 상황 판단"""
        score_diff = abs(game_state.get('score_diff', 0))
        inning = game_state.get('inning', 1)
        runners = sum(game_state.get('runners', [0, 0, 0]))
        count = game_state.get('count', '0-0')

        return (
            score_diff <= cls.LEVERAGE_THRESHOLDS['high_leverage_score_diff'] or
            inning >= cls.LEVERAGE_THRESHOLDS['high_leverage_inning'] or
            runners >= cls.LEVERAGE_THRESHOLDS['high_leverage_runners'] or
            count in cls.LEVERAGE_THRESHOLDS['critical_count']
        )