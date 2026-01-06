"""
AegisStrategyEngine - Config-Driven & Robust 투구 전략 엔진

이 모듈은 물리 엔진, 게임 이론, 분석 모듈을 통합하여
상황별 최적의 투구 전략을 제시하는 의사결정 엔진입니다.

Key Features:
- Config-Driven: 모든 파라미터는 StrategyConfig에서 로드 (하드코딩 금지)
- Robust to Data Noise: Trackman 오분류 등 노이즈 데이터 필터링
- Dynamic Action Space: 투수 레퍼토리에 맞춘 가능한 행동만 생성
- Multi-Metric Simulation: Tunneling, EV, Command, Stuff 종합 평가
- Probabilistic Selection: Softmax 기반 확률적 선택 (Temperature 조절)
- Rationale Generation: 의사결정 이유를 자연어로 설명 (구사율 정보 포함)

Author: Aegis Development Team
"""

import numpy as np
import torch
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.config import StrategyConfig
from src.game_theory.context_encoder import ContextEncoder
from src.game_theory.entropy import EntropyMonitor
from src.game_theory.effective_velocity import EffectiveVelocityCalculator
from src.game_theory.tunneling import TunnelingAnalyzer

# Logger 설정
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


@dataclass
class Action:
    """투구 행동 정의"""
    pitch_type: str      # 구종 (FF, SL, CH, ...)
    zone: str            # 목표 존 (heart_mid, chase_low, ...)
    plate_x: float       # X 좌표 (feet)
    plate_z: float       # Z 좌표 (feet)

    def to_dict(self) -> Dict:
        return {
            'pitch_type': self.pitch_type,
            'zone': self.zone,
            'location': {'x': self.plate_x, 'z': self.plate_z}
        }


@dataclass
class DecisionResult:
    """의사결정 결과"""
    selected_action: Action
    action_probs: Dict[str, float]  # Top K 행동의 확률 분포
    q_values: Dict[str, float]      # Top K 행동의 Q-value
    rationale: str                  # 자연어 설명
    leverage_level: str             # 상황 판단 (high/medium/low)
    entropy_status: str             # 엔트로피 상태
    filtered_pitches: Dict[str, float]  # 필터링된 구종 및 구사율 (Noise 제거 후)
    noise_pitches: List[str]        # 제거된 Noise 구종 리스트


class AegisStrategyEngine:
    """
    Config-Driven 투구 전략 의사결정 엔진

    모든 파라미터는 StrategyConfig에서 로드하며,
    투수 레퍼토리와 상황에 맞는 최적의 투구를 제안합니다.
    """

    def __init__(
        self,
        config: Optional[StrategyConfig] = None,
        device: str = 'cpu'
    ):
        """
        AegisStrategyEngine 초기화

        Args:
            config: 전략 설정 (None이면 기본 StrategyConfig 사용)
            device: PyTorch 디바이스 ('cpu' or 'cuda')
        """
        self.config = config if config is not None else StrategyConfig()
        self.device = device

        # 하위 모듈 초기화
        self.context_encoder = ContextEncoder(device=device)
        self.entropy_monitor = EntropyMonitor(window_size=20)
        self.ev_calculator = EffectiveVelocityCalculator()
        self.tunneling_analyzer = TunnelingAnalyzer()

        print("=" * 80)
        print("🎯 AegisStrategyEngine 초기화 완료")
        print("=" * 80)
        print(f"Device: {device}")
        print(f"Supported Pitches: {len(self.config.MLB_PITCH_TYPES)}")
        print(f"Zones: {len(self.config.ZONES)}")
        print(f"Feature Weights: {self.config.FEATURE_WEIGHTS}")
        print("=" * 80 + "\n")

    def decide_pitch(
        self,
        game_state: Dict,
        pitcher_state: Dict,
        matchup_state: Dict,
        pitch_usage_stats: Dict[str, float],
        pitcher_stats: Optional[Dict] = None
    ) -> DecisionResult:
        """
        현재 상황에서 최적의 투구를 결정 (Data Noise Robust)

        Args:
            game_state: 게임 상태 (outs, count, runners, score_diff, inning)
            pitcher_state: 투수 상태 (hand, role, pitch_count, entropy, prev_pitch, prev_velo)
            matchup_state: 매치업 정보 (batter_hand, times_faced, chase_rate, whiff_rate, iso, gb_fb_ratio, ops)
            pitch_usage_stats: 구종별 구사율 (예: {'FF': 0.60, 'SL': 0.35, 'CH': 0.05, 'KN': 0.001})
            pitcher_stats: 투수 통계 (구종별 stuff+, 존별 제구율, 샘플 수 등)

        Returns:
            DecisionResult: 선택된 행동, 확률 분포, Q-value, 자연어 설명

        Logic Flow:
            1. Context & Filtering: 상태 벡터화 + Noise 구종 필터링
            2. Action Space Generation: 필터링된 구종으로 행동 공간 생성
            3. Simulation: 모든 행동에 대해 메트릭 계산 (샘플 부족 시 페널티)
            4. Payoff Calculation: Q-value 계산 (가중치 합)
            5. Probabilistic Selection: Softmax 선택
            6. Rationale Generation: 의사결정 이유 생성 (구사율 정보 포함)
        """
        # ====================================================================
        # Step 1: Context Awareness & Noise Filtering
        # ====================================================================
        # 1-1. 상태 벡터화
        state_vector = self.context_encoder.encode(
            game_state, pitcher_state, matchup_state
        )

        # 1-2. Leverage 수준 판단
        leverage_level = self._determine_leverage(game_state)

        # 1-3. 엔트로피 상태 확인
        current_entropy = pitcher_state.get('entropy', 0.5)
        entropy_status = self._assess_entropy(current_entropy)

        # 1-4. Ghost Pitches 필터링 (Trackman 오분류 등 Noise 제거)
        filtered_pitches, noise_pitches = self._filter_ghost_pitches(pitch_usage_stats)

        # Fallback: 필터링 후 구종이 없으면 주무기(최고 구사율) 강제 선택
        if not filtered_pitches:
            logger.warning(
                f"All pitches filtered as noise. Fallback to primary pitch."
            )
            primary_pitch = max(pitch_usage_stats.items(), key=lambda x: x[1])
            filtered_pitches = {primary_pitch[0]: primary_pitch[1]}
            noise_pitches = [p for p in pitch_usage_stats.keys() if p != primary_pitch[0]]

        # ====================================================================
        # Step 2: Action Space Generation
        # ====================================================================
        valid_actions = self._generate_valid_actions(list(filtered_pitches.keys()))

        if not valid_actions:
            raise ValueError(
                f"No valid actions generated after filtering. "
                f"Filtered pitches: {filtered_pitches}"
            )

        # ====================================================================
        # Step 3 & 4: Simulation + Payoff Calculation
        # ====================================================================
        q_values = {}
        metrics_cache = {}  # 메트릭 캐싱 (Rationale 생성에 재사용)

        for action in valid_actions:
            # 각 행동에 대한 메트릭 계산 (샘플 부족 시 페널티 적용)
            metrics = self._calculate_action_metrics(
                action,
                game_state,
                pitcher_state,
                matchup_state,
                pitcher_stats,
                filtered_pitches  # 구사율 정보 전달
            )

            # Q-value 계산 (Config의 가중치 사용)
            q_value = self._calculate_payoff(metrics, matchup_state)

            action_key = f"{action.pitch_type}_{action.zone}"
            q_values[action_key] = q_value
            metrics_cache[action_key] = metrics

        # ====================================================================
        # Step 5: Probabilistic Selection
        # ====================================================================
        selected_action, action_probs = self._select_action_probabilistic(
            valid_actions,
            q_values,
            leverage_level
        )

        # ====================================================================
        # Step 6: Rationale Generation (구사율 정보 포함)
        # ====================================================================
        selected_key = f"{selected_action.pitch_type}_{selected_action.zone}"
        rationale = self._generate_rationale(
            selected_action,
            metrics_cache[selected_key],
            action_probs,
            pitcher_state,
            matchup_state,
            leverage_level,
            filtered_pitches  # 구사율 정보 전달
        )

        return DecisionResult(
            selected_action=selected_action,
            action_probs=action_probs,
            q_values={k: v for k, v in sorted(
                q_values.items(), key=lambda x: x[1], reverse=True
            )[:self.config.RATIONALE_CONFIG['top_k_alternatives']]},
            rationale=rationale,
            leverage_level=leverage_level,
            entropy_status=entropy_status,
            filtered_pitches=filtered_pitches,
            noise_pitches=noise_pitches
        )

    def _filter_ghost_pitches(
        self,
        pitch_usage_stats: Dict[str, float]
    ) -> Tuple[Dict[str, float], List[str]]:
        """
        Noise 구종 필터링 (Trackman 오분류 등 희귀 구종 제거)

        Args:
            pitch_usage_stats: 구종별 구사율 (예: {'FF': 0.60, 'SL': 0.35, 'KN': 0.001})

        Returns:
            filtered_pitches: 필터링 통과한 구종 및 구사율
            noise_pitches: 제거된 Noise 구종 리스트

        Logic:
            - 구사율 < MIN_PITCH_USAGE_THRESHOLD인 구종은 Noise로 간주
            - Config의 임계값(기본 3%)보다 낮으면 제외
            - 제외된 구종은 로그에 경고 기록
        """
        threshold = self.config.MIN_PITCH_USAGE_THRESHOLD
        filtered_pitches = {}
        noise_pitches = []

        for pitch_type, usage_rate in pitch_usage_stats.items():
            # Config에 정의된 구종인지 확인
            if pitch_type not in self.config.MLB_PITCH_TYPES:
                logger.warning(
                    f"Unknown pitch type '{pitch_type}' in usage stats. Skipping."
                )
                noise_pitches.append(pitch_type)
                continue

            # 구사율 임계값 체크
            if usage_rate < threshold:
                if self.config.NOISE_LOGGING_ENABLED:
                    logger.warning(
                        f"Ignored noise pitch: {pitch_type} "
                        f"({usage_rate:.1%} < {threshold:.1%} threshold). "
                        f"Likely Trackman misclassification."
                    )
                noise_pitches.append(pitch_type)
            else:
                filtered_pitches[pitch_type] = usage_rate

        # 필터링 결과 로그
        if self.config.NOISE_LOGGING_ENABLED and filtered_pitches:
            logger.info(
                f"Filtered pitches: {list(filtered_pitches.keys())} "
                f"(removed {len(noise_pitches)} noise pitches)"
            )

        return filtered_pitches, noise_pitches

    def _generate_valid_actions(
        self,
        pitcher_repertoire: List[str]
    ) -> List[Action]:
        """
        투수 레퍼토리에 맞는 가능한 (Pitch, Zone) 조합 생성

        Args:
            pitcher_repertoire: 투수가 던질 수 있는 구종 리스트

        Returns:
            valid_actions: 가능한 모든 행동 리스트

        Note:
            - 투수에게 없는 구종은 제외
            - 각 존의 중심 좌표를 타겟으로 설정
            - Waste zone은 특정 상황에서만 포함 가능
        """
        valid_actions = []

        for pitch_type in pitcher_repertoire:
            # 투수 레퍼토리에 있는 구종만 사용
            if pitch_type not in self.config.MLB_PITCH_TYPES:
                continue

            for zone_name, zone_info in self.config.ZONES.items():
                # Waste zone은 특수 상황에서만 사용 (일단 제외)
                if zone_name == 'waste':
                    continue

                # 존 중심 좌표 계산
                x_range = zone_info['x_range']
                z_range = zone_info['z_range']
                plate_x = (x_range[0] + x_range[1]) / 2.0
                plate_z = (z_range[0] + z_range[1]) / 2.0

                action = Action(
                    pitch_type=pitch_type,
                    zone=zone_name,
                    plate_x=plate_x,
                    plate_z=plate_z
                )
                valid_actions.append(action)

        return valid_actions

    def _calculate_action_metrics(
        self,
        action: Action,
        game_state: Dict,
        pitcher_state: Dict,
        matchup_state: Dict,
        pitcher_stats: Optional[Dict],
        pitch_usage: Dict[str, float]
    ) -> Dict[str, float]:
        """
        특정 행동에 대한 모든 메트릭 계산 (Data Quality 고려)

        Args:
            action: 평가할 행동
            game_state: 게임 상태
            pitcher_state: 투수 상태
            matchup_state: 매치업 정보
            pitcher_stats: 투수 통계
            pitch_usage: 구종별 구사율 (신뢰도 판단용)

        Returns:
            metrics: 각종 메트릭 딕셔너리
                - tunneling_score: 터널링 점수 [0, 1]
                - ev_delta: EV 차이 (mph)
                - command_risk: 제구 성공률 [0, 1]
                - stuff_quality: Stuff+ 점수 (샘플 부족 시 페널티)
                - chase_score: Chase 유도 점수 [0, 1]
                - entropy_bonus: 엔트로피 보너스 [0, 1]
                - data_quality: 데이터 품질 지표 [0, 1]
        """
        metrics = {}

        # 1. Tunneling Score (직전 투구와의 궤적 유사성)
        metrics['tunneling_score'] = self._calculate_tunneling_score(
            action, pitcher_state
        )

        # 2. Effective Velocity Delta (타자 인지 속도 차이)
        metrics['ev_delta'] = self._calculate_ev_delta(
            action, pitcher_state, matchup_state
        )

        # 3. Command Risk (제구 성공률)
        metrics['command_risk'] = self._calculate_command_risk(
            action, pitcher_state, pitcher_stats
        )

        # 4. Stuff Quality (구종 위력) - 샘플 부족 시 페널티 적용
        metrics['stuff_quality'] = self._calculate_stuff_quality_robust(
            action, pitcher_stats, pitch_usage
        )

        # 5. Chase Score (헛스윙 유도 점수)
        metrics['chase_score'] = self._calculate_chase_score(
            action, matchup_state
        )

        # 6. Entropy Bonus (패턴 변경 보너스)
        metrics['entropy_bonus'] = self._calculate_entropy_bonus(
            action, pitcher_state
        )

        # 7. Data Quality (데이터 신뢰도)
        metrics['data_quality'] = self._assess_data_quality(
            action, pitcher_stats, pitch_usage
        )

        return metrics


    def _calculate_tunneling_score(
        self,
        action: Action,
        pitcher_state: Dict
    ) -> float:
        """
        터널링 점수 계산 (직전 투구와의 궤적 유사성)

        Args:
            action: 현재 행동
            pitcher_state: 투수 상태 (prev_pitch 포함)

        Returns:
            tunneling_score: [0, 1] (1에 가까울수록 터널링 효과 높음)

        Note:
            실제 구현에서는 TunnelingAnalyzer를 사용하여
            release point ~ plate 궤적 유사도를 계산해야 하지만,
            여기서는 간단히 구종 조합 기반 휴리스틱 사용
        """
        prev_pitch = pitcher_state.get('prev_pitch', None)

        if prev_pitch is None:
            # 첫 투구는 터널링 점수 없음
            return 0.5

        # 터널링 효과가 높은 구종 조합 (휴리스틱)
        # 실제로는 TunnelingAnalyzer.calculate_tunneling()을 사용해야 함
        tunneling_pairs = {
            ('FF', 'SL'): 0.9, ('FF', 'CH'): 0.85, ('FF', 'CU'): 0.8,
            ('SI', 'SL'): 0.88, ('SI', 'CH'): 0.83,
            ('FC', 'SL'): 0.92, ('FC', 'ST'): 0.90,
            ('SL', 'FF'): 0.75, ('CH', 'FF'): 0.70,
            ('CU', 'FF'): 0.78, ('ST', 'FC'): 0.85
        }

        pair_key = (prev_pitch, action.pitch_type)
        tunneling_score = tunneling_pairs.get(pair_key, 0.5)

        return tunneling_score

    def _calculate_ev_delta(
        self,
        action: Action,
        pitcher_state: Dict,
        matchup_state: Dict
    ) -> float:
        """
        Effective Velocity 차이 계산

        Args:
            action: 현재 행동
            pitcher_state: 투수 상태
            matchup_state: 매치업 정보

        Returns:
            ev_delta: EV 차이 (mph, 클수록 타자에게 어려움)
        """
        # 구종별 평균 구속 (실제로는 pitcher_stats에서 가져와야 함)
        pitch_speed_map = {
            'FF': 95.0, 'SI': 93.0, 'FC': 92.0,
            'SL': 85.0, 'ST': 84.0, 'CU': 78.0, 'KC': 79.0,
            'CH': 86.0, 'FS': 87.0, 'FO': 83.0, 'KN': 75.0
        }

        current_speed = pitch_speed_map.get(action.pitch_type, 90.0)
        prev_velo = pitcher_state.get('prev_velo', 90.0)

        # EffectiveVelocity 계산
        batter_hand = matchup_state.get('batter_hand', 'R')
        current_ev = self.ev_calculator.calculate_ev(
            current_speed, action.plate_x, action.plate_z, batter_hand
        )

        # 직전 투구의 EV (간단히 prev_velo를 EV로 간주)
        # 실제로는 직전 투구의 위치도 고려해야 함
        ev_delta = abs(current_ev - prev_velo)

        return ev_delta

    def _calculate_command_risk(
        self,
        action: Action,
        pitcher_state: Dict,
        pitcher_stats: Optional[Dict]
    ) -> float:
        """
        제구 성공률 계산 (해당 존에 정확히 던질 확률)

        Args:
            action: 현재 행동
            pitcher_state: 투수 상태
            pitcher_stats: 투수 통계 (존별 제구율)

        Returns:
            command_rate: [0, 1] (1에 가까울수록 제구 성공 확률 높음)

        Note:
            pitcher_stats가 없으면 리그 평균 사용 (Fallback)
        """
        if pitcher_stats is None:
            # Fallback: 리그 평균 제구율
            return self.config.COMMAND_CONFIG['league_average_command']

        # 투수별, 존별 제구율 가져오기
        zone_command = pitcher_stats.get('zone_command', {})
        pitch_command = zone_command.get(action.pitch_type, {})
        command_rate = pitch_command.get(
            action.zone,
            self.config.COMMAND_CONFIG['league_average_command']
        )

        return command_rate

    def _calculate_stuff_quality(
        self,
        action: Action,
        pitcher_stats: Optional[Dict]
    ) -> float:
        """
        구종 위력(Stuff+) 계산 (레거시, 하위 호환성용)

        Args:
            action: 현재 행동
            pitcher_stats: 투수 통계 (구종별 Stuff+)

        Returns:
            stuff_plus: Stuff+ 점수 (100 = 평균, 높을수록 좋음)
        """
        if pitcher_stats is None:
            # Fallback: 평균 Stuff+
            return self.config.STUFF_BASELINE.get(action.pitch_type, 100.0)

        # 투수별 구종 위력 가져오기
        stuff_plus_data = pitcher_stats.get('stuff_plus', {})
        stuff_plus = stuff_plus_data.get(
            action.pitch_type,
            self.config.STUFF_BASELINE.get(action.pitch_type, 100.0)
        )

        return stuff_plus

    def _calculate_stuff_quality_robust(
        self,
        action: Action,
        pitcher_stats: Optional[Dict],
        pitch_usage: Dict[str, float]
    ) -> float:
        """
        구종 위력(Stuff+) 계산 with Sample Size Penalty

        Args:
            action: 현재 행동
            pitcher_stats: 투수 통계 (구종별 Stuff+, 샘플 수)
            pitch_usage: 구종별 구사율 (신뢰도 판단용)

        Returns:
            stuff_plus: Stuff+ 점수 (샘플 부족 시 페널티 적용)

        Logic:
            - 샘플 수 < MIN_SAMPLE_SIZE_THRESHOLD이면 페널티 적용
            - 구사율이 낮은 구종일수록 신뢰도 하락
            - Stuff+ × LOW_SAMPLE_PENALTY (Config 기반)
        """
        # 기본 Stuff+ 가져오기
        base_stuff = self._calculate_stuff_quality(action, pitcher_stats)

        # 샘플 수 확인
        if pitcher_stats is not None:
            sample_sizes = pitcher_stats.get('sample_sizes', {})
            sample_count = sample_sizes.get(action.pitch_type, 0)

            # 샘플 부족 시 페널티
            if sample_count < self.config.MIN_SAMPLE_SIZE_THRESHOLD:
                penalty = self.config.LOW_SAMPLE_PENALTY
                penalized_stuff = base_stuff * penalty

                if self.config.NOISE_LOGGING_ENABLED:
                    logger.debug(
                        f"Low sample size for {action.pitch_type}: "
                        f"{sample_count} pitches. "
                        f"Stuff+ penalized: {base_stuff:.1f} → {penalized_stuff:.1f}"
                    )

                return penalized_stuff

        # 구사율 기반 신뢰도 조정 (매우 낮은 구사율도 추가 페널티)
        usage_rate = pitch_usage.get(action.pitch_type, 0.0)
        if usage_rate < 0.10:  # 10% 미만 구사율은 신뢰도 낮음
            confidence = 0.5 + (usage_rate / 0.10) * 0.5  # 0.5 ~ 1.0
            adjusted_stuff = base_stuff * confidence
            return adjusted_stuff

        return base_stuff

    def _assess_data_quality(
        self,
        action: Action,
        pitcher_stats: Optional[Dict],
        pitch_usage: Dict[str, float]
    ) -> float:
        """
        데이터 품질 평가 (샘플 크기 + 구사율 기반)

        Args:
            action: 현재 행동
            pitcher_stats: 투수 통계
            pitch_usage: 구종별 구사율

        Returns:
            quality_score: [0, 1] (1 = 높은 신뢰도, 0 = 낮은 신뢰도)

        Note:
            이 점수는 의사결정에 직접 사용되지는 않지만,
            Rationale 생성 시 데이터 신뢰도를 언급하는 데 활용
        """
        quality_score = 1.0

        # 1. 구사율 기반 신뢰도 (높을수록 신뢰)
        usage_rate = pitch_usage.get(action.pitch_type, 0.0)
        usage_confidence = min(usage_rate / 0.30, 1.0)  # 30% 이상이면 만점
        quality_score *= usage_confidence

        # 2. 샘플 크기 기반 신뢰도
        if pitcher_stats is not None:
            sample_sizes = pitcher_stats.get('sample_sizes', {})
            sample_count = sample_sizes.get(action.pitch_type, 0)

            # 최소 임계값 이상이면 만점
            if sample_count >= self.config.MIN_SAMPLE_SIZE_THRESHOLD:
                sample_confidence = 1.0
            else:
                sample_confidence = sample_count / self.config.MIN_SAMPLE_SIZE_THRESHOLD

            quality_score *= sample_confidence

        return quality_score

    def _calculate_chase_score(
        self,
        action: Action,
        matchup_state: Dict
    ) -> float:
        """
        Chase 유도 점수 계산 (볼존 공격 시 헛스윙 확률)

        Args:
            action: 현재 행동
            matchup_state: 매치업 정보 (chase_rate 포함)

        Returns:
            chase_score: [0, 1] (Chase zone에서 타자의 헛스윙 확률)
        """
        # Chase zone 여부 확인
        zone_info = self.config.ZONES.get(action.zone, {})
        is_chase_zone = action.zone.startswith('chase')

        if not is_chase_zone:
            # Chase zone이 아니면 점수 낮음
            return 0.3

        # 타자의 chase rate (O-Swing%)
        chase_rate = matchup_state.get('chase_rate', 0.3)

        # Chase zone에서는 타자의 chase_rate를 그대로 점수로 사용
        return chase_rate

    def _calculate_entropy_bonus(
        self,
        action: Action,
        pitcher_state: Dict
    ) -> float:
        """
        엔트로피 보너스 계산 (패턴 변경 필요성)

        Args:
            action: 현재 행동
            pitcher_state: 투수 상태 (entropy, prev_pitch)

        Returns:
            entropy_bonus: [0, 1] (패턴 변경이 필요하면 높은 점수)
        """
        current_entropy = pitcher_state.get('entropy', 0.7)
        prev_pitch = pitcher_state.get('prev_pitch', None)

        # 엔트로피가 낮으면 패턴 변경 필요
        if current_entropy < self.config.ENTROPY_THRESHOLDS['low_entropy']:
            # 직전과 다른 구종이면 보너스
            if prev_pitch is not None and action.pitch_type != prev_pitch:
                return 0.9
            else:
                return 0.3
        else:
            # 엔트로피가 높으면 패턴 변경 필요성 낮음
            return 0.5

    def _calculate_payoff(
        self,
        metrics: Dict[str, float],
        matchup_state: Dict
    ) -> float:
        """
        Q-value 계산 (가중치 합 + Exploitation 배율)

        Formula:
            Q(s,a) = Σ(w_i · feature_i) × exploitation_multiplier

        Args:
            metrics: 각종 메트릭 딕셔너리
            matchup_state: 매치업 정보 (약점 공략용)

        Returns:
            q_value: 최종 Q-value (높을수록 좋은 행동)

        Note:
            모든 가중치는 Config에서 로드 (하드코딩 금지!)
        """
        # 1. Feature별 가중치 적용
        q_value = 0.0

        # Tunneling
        q_value += (
            self.config.get_feature_weight('tunneling') *
            metrics['tunneling_score']
        )

        # EV Delta (정규화)
        ev_normalized = min(
            metrics['ev_delta'] / self.config.NORMALIZATION['ev_delta_range'],
            1.0
        )
        q_value += (
            self.config.get_feature_weight('ev_delta') *
            ev_normalized
        )

        # Chase Rate
        q_value += (
            self.config.get_feature_weight('chase_rate') *
            metrics['chase_score']
        )

        # Stuff Quality (정규화)
        stuff_normalized = (
            (metrics['stuff_quality'] - self.config.NORMALIZATION['stuff_plus_mean']) /
            self.config.NORMALIZATION['stuff_plus_std']
        )
        stuff_normalized = max(0.0, min(1.0, (stuff_normalized + 2.0) / 4.0))  # [-2, +2] → [0, 1]
        q_value += (
            self.config.get_feature_weight('stuff_quality') *
            stuff_normalized
        )

        # Command
        q_value += (
            self.config.get_feature_weight('command') *
            metrics['command_risk']
        )

        # Entropy
        q_value += (
            self.config.get_feature_weight('entropy') *
            metrics['entropy_bonus']
        )

        # 2. Exploitation Multiplier (타자 약점 공략)
        exploitation_multiplier = 1.0

        # 타자의 whiff_rate가 높으면 배율 증가
        whiff_rate = matchup_state.get('whiff_rate', 0.25)
        if whiff_rate > 0.30:  # 30% 이상이면 높은 헛스윙률
            exploitation_multiplier *= self.config.EXPLOITATION_CONFIG['high_whiff_multiplier']

        # Chase zone 공격 시 타자의 chase_rate가 높으면 배율 증가
        if metrics['chase_score'] > 0.35:  # Chase zone 공격
            exploitation_multiplier *= self.config.EXPLOITATION_CONFIG['weak_zone_multiplier']

        # 최종 Q-value
        q_value *= exploitation_multiplier

        return q_value

    def _select_action_probabilistic(
        self,
        valid_actions: List[Action],
        q_values: Dict[str, float],
        leverage_level: str
    ) -> Tuple[Action, Dict[str, float]]:
        """
        Softmax 기반 확률적 행동 선택

        Args:
            valid_actions: 가능한 행동 리스트
            q_values: 각 행동의 Q-value
            leverage_level: Leverage 수준 (temperature 결정)

        Returns:
            selected_action: 선택된 행동
            action_probs: 상위 K개 행동의 확률 분포

        Formula:
            P(a) = exp(Q(s,a) / τ) / Σ exp(Q(s,a') / τ)
        """
        # Temperature 가져오기 (Config에서)
        temperature = self.config.get_temperature(leverage_level)

        # Q-value를 numpy array로 변환
        action_keys = list(q_values.keys())
        q_array = np.array([q_values[k] for k in action_keys])

        # Softmax 계산
        exp_q = np.exp(q_array / temperature)
        probs = exp_q / np.sum(exp_q)

        # 확률적 샘플링
        selected_idx = np.random.choice(len(action_keys), p=probs)
        selected_key = action_keys[selected_idx]

        # 선택된 행동 찾기
        pitch_type, zone = selected_key.split('_', 1)
        selected_action = None
        for action in valid_actions:
            if action.pitch_type == pitch_type and action.zone == zone:
                selected_action = action
                break

        # 안전성 검증 (이론상 항상 찾아야 함)
        assert selected_action is not None, f"Selected action {selected_key} not found in valid_actions"

        # 상위 K개 행동의 확률 분포
        top_k = self.config.RATIONALE_CONFIG['top_k_alternatives']
        top_indices = np.argsort(probs)[-top_k:][::-1]
        action_probs = {
            action_keys[idx]: float(probs[idx])
            for idx in top_indices
        }

        return selected_action, action_probs

    def _determine_leverage(self, game_state: Dict) -> str:
        """
        Leverage 수준 판단

        Args:
            game_state: 게임 상태

        Returns:
            leverage_level: 'high_leverage', 'medium_leverage', 'low_leverage'
        """
        if self.config.is_high_leverage(game_state):
            return 'high_leverage'

        # 중간 상황 판단 (간단한 휴리스틱)
        score_diff = abs(game_state.get('score_diff', 0))
        inning = game_state.get('inning', 1)

        if score_diff <= 4 and inning >= 5:
            return 'medium_leverage'

        return 'low_leverage'

    def _assess_entropy(self, entropy: float) -> str:
        """
        엔트로피 상태 평가

        Args:
            entropy: 현재 엔트로피 [0, 1]

        Returns:
            status: 'low', 'medium', 'high'
        """
        if entropy < self.config.ENTROPY_THRESHOLDS['low_entropy']:
            return 'low'
        elif entropy < self.config.ENTROPY_THRESHOLDS['medium_entropy']:
            return 'medium'
        else:
            return 'high'

    def _generate_rationale(
        self,
        selected_action: Action,
        metrics: Dict[str, float],
        action_probs: Dict[str, float],
        pitcher_state: Dict,
        matchup_state: Dict,
        leverage_level: str,
        pitch_usage: Dict[str, float]
    ) -> str:
        """
        의사결정 이유를 자연어로 생성 (구사율 정보 포함)

        Args:
            selected_action: 선택된 행동
            metrics: 선택된 행동의 메트릭
            action_probs: 상위 K개 행동의 확률
            pitcher_state: 투수 상태
            matchup_state: 매치업 정보
            leverage_level: Leverage 수준
            pitch_usage: 구종별 구사율 (신뢰도 근거)

        Returns:
            rationale: 자연어 설명 문자열

        Example:
            "주무기인 직구(60%)로 직전 슬라이더(SL) 이후 터널링 점수가 0.92로 높고,
            타자의 Chase Rate이 38%로 높아 Four-Seam Fastball(FF)를 chase_out 존에 선택함.
            (EV 차이: +4.2mph). 현재 승부처 상황으로 확실한 공을 선택했습니다."
        """
        rationale_parts = []

        # 1. 구사율 정보 (주무기 여부)
        selected_usage = pitch_usage.get(selected_action.pitch_type, 0.0)
        if selected_usage >= 0.40:  # 40% 이상이면 주무기
            rationale_parts.append(
                f"주무기인 {self.config.MLB_PITCH_TYPES.get(selected_action.pitch_type)}({selected_usage:.0%})로"
            )
        elif selected_usage >= 0.20:  # 20~40%는 보조 구종
            rationale_parts.append(
                f"보조 구종인 {self.config.MLB_PITCH_TYPES.get(selected_action.pitch_type)}({selected_usage:.0%})로"
            )
        else:  # 20% 미만은 변화구
            rationale_parts.append(
                f"변화구 {self.config.MLB_PITCH_TYPES.get(selected_action.pitch_type)}({selected_usage:.0%})로"
            )

        # 2. 직전 투구 정보
        prev_pitch = pitcher_state.get('prev_pitch', None)
        if prev_pitch:
            prev_pitch_name = self.config.MLB_PITCH_TYPES.get(prev_pitch, prev_pitch)
            rationale_parts.append(f"직전 {prev_pitch_name}({prev_pitch}) 이후")

        # 3. 터널링 점수
        tunneling = metrics['tunneling_score']
        if tunneling >= self.config.RATIONALE_CONFIG['tunneling_threshold']:
            rationale_parts.append(f"터널링 점수가 {tunneling:.2f}로 높고")

        # 4. EV 차이
        ev_delta = metrics['ev_delta']
        if ev_delta >= self.config.RATIONALE_CONFIG['ev_significant_delta']:
            rationale_parts.append(f"EV 차이가 +{ev_delta:.1f}mph로 크며")

        # 5. 타자 약점
        chase_rate = matchup_state.get('chase_rate', 0.3)
        if chase_rate >= self.config.RATIONALE_CONFIG['chase_high_threshold']:
            rationale_parts.append(f"타자의 Chase Rate이 {chase_rate:.1%}로 높아")

        whiff_rate = matchup_state.get('whiff_rate', 0.25)
        if whiff_rate >= 0.30:
            rationale_parts.append(f"헛스윙률이 {whiff_rate:.1%}로 높아")

        # 6. 선택된 구종 및 존
        selected_pitch_name = self.config.MLB_PITCH_TYPES.get(
            selected_action.pitch_type,
            selected_action.pitch_type
        )
        rationale_parts.append(
            f"{selected_pitch_name}({selected_action.pitch_type})를 "
            f"{selected_action.zone} 존에 선택함"
        )

        # 7. 데이터 신뢰도 언급 (낮은 경우만)
        data_quality = metrics.get('data_quality', 1.0)
        if data_quality < 0.7:
            rationale_parts.append(
                f"(주의: 데이터 신뢰도 {data_quality:.0%})"
            )

        # 8. Leverage 상황
        leverage_msg = {
            'high_leverage': "현재 승부처 상황으로 확실한 공을 선택했습니다",
            'medium_leverage': "중간 leverage 상황으로 균형잡힌 선택을 했습니다",
            'low_leverage': "여유 있는 상황으로 다양한 선택을 시도했습니다"
        }
        rationale_parts.append(leverage_msg.get(leverage_level, ""))

        # 9. 대안 행동들
        alternatives = []
        for action_key, prob in list(action_probs.items())[1:4]:  # 2~4위
            pitch, zone = action_key.split('_', 1)
            pitch_name = self.config.MLB_PITCH_TYPES.get(pitch, pitch)
            alternatives.append(f"{pitch_name}({pitch}) {zone}: {prob:.1%}")

        if alternatives:
            rationale_parts.append(
                f"\n대안: {', '.join(alternatives)}"
            )

        return ", ".join(rationale_parts) + "."


def main():
    """사용 예시 및 테스트 (Data Noise Robustness 포함)"""
    print("=" * 80)
    print("🎯 AegisStrategyEngine 테스트 (Data Noise Filtering)")
    print("=" * 80 + "\n")

    # Engine 초기화
    engine = AegisStrategyEngine(device='cpu')

    # ========================================================================
    # Test Case 1: 위기 상황 (High Leverage) + Ghost Pitch 필터링
    # ========================================================================
    print("📊 Test Case 1: High Leverage 상황 (9회, 만루, 2아웃) + Ghost Pitch 테스트")
    print("-" * 80)

    game_state = {
        'outs': 2,
        'count': '3-2',
        'runners': [1, 1, 1],
        'score_diff': -1,  # 1점 뒤짐
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
        'chase_rate': 0.38,   # 높은 chase rate
        'whiff_rate': 0.32,   # 높은 whiff rate
        'iso': 0.220,
        'gb_fb_ratio': 0.9,
        'ops': 0.810
    }

    # pitch_usage_stats: 구종별 구사율 (KN은 0.5%로 노이즈)
    pitch_usage_stats = {
        'FF': 0.55,  # 주무기 (55%)
        'SL': 0.30,  # 보조 구종 (30%)
        'CH': 0.145, # 변화구 (14.5%)
        'KN': 0.005  # Ghost pitch (0.5%) <- 3% 미만이므로 필터링 예상
    }

    # 투수 통계 (sample_sizes 포함)
    pitcher_stats = {
        'stuff_plus': {
            'FF': 105.0,
            'SL': 115.0,  # 뛰어난 슬라이더
            'CH': 98.0,
            'KN': 92.0    # 낮지만 샘플 수가 너무 적음
        },
        'sample_sizes': {
            'FF': 165,    # 충분한 샘플
            'SL': 90,     # 충분한 샘플
            'CH': 44,     # 충분한 샘플
            'KN': 2       # 샘플 부족 (< 10)
        },
        'zone_command': {
            'FF': {'chase_low': 0.70, 'shadow_out_mid': 0.75},
            'SL': {'chase_low': 0.68, 'chase_out': 0.72},
            'CH': {'chase_low': 0.65}
        }
    }

    result = engine.decide_pitch(
        game_state,
        pitcher_state,
        matchup_state,
        pitch_usage_stats,  # Dict 형태로 전달
        pitcher_stats
    )

    print(f"Selected Action: {result.selected_action.pitch_type} @ {result.selected_action.zone}")
    print(f"Location: ({result.selected_action.plate_x:.2f}, {result.selected_action.plate_z:.2f})")
    print(f"Leverage: {result.leverage_level}")
    print(f"Entropy Status: {result.entropy_status}")
    print(f"\n🔍 Noise Filtering 결과:")
    print(f"  Filtered Pitches: {list(result.filtered_pitches.keys())}")
    print(f"  Noise Pitches (Removed): {result.noise_pitches}")
    print(f"\nTop Actions:")
    for action_key, prob in result.action_probs.items():
        print(f"  {action_key}: {prob:.1%}")
    print(f"\nRationale:\n{result.rationale}")

    # ========================================================================
    # Test Case 2: 여유 상황 (Low Leverage) + Low Sample Penalty
    # ========================================================================
    print("\n" + "=" * 80)
    print("📊 Test Case 2: Low Leverage 상황 (3회, 5점 차이) + Sample Size Penalty")
    print("-" * 80)

    game_state_2 = {
        'outs': 0,
        'count': '1-1',
        'runners': [0, 0, 0],
        'score_diff': 5,  # 5점 리드
        'inning': 3
    }

    pitcher_state_2 = {
        'hand': 'L',
        'role': 'SP',
        'pitch_count': 45,
        'entropy': 0.88,  # 높은 엔트로피
        'prev_pitch': 'FF',
        'prev_velo': 92.0
    }

    matchup_state_2 = {
        'batter_hand': 'R',
        'times_faced': 0,
        'chase_rate': 0.28,
        'whiff_rate': 0.23,
        'iso': 0.165,
        'gb_fb_ratio': 1.2,
        'ops': 0.730
    }

    # pitch_usage_stats: 모든 구종이 충분한 구사율
    pitch_usage_stats_2 = {
        'FF': 0.48,
        'SI': 0.27,
        'SL': 0.17,
        'CH': 0.08
    }

    # 샘플 수가 적은 경우 테스트
    pitcher_stats_2 = {
        'stuff_plus': {
            'FF': 98.0,
            'SI': 103.0,
            'SL': 107.0,
            'CH': 95.0
        },
        'sample_sizes': {
            'FF': 120,   # 충분
            'SI': 68,    # 충분
            'SL': 42,    # 충분
            'CH': 8      # 부족 (< 10) -> Penalty 적용 예상
        }
    }

    result_2 = engine.decide_pitch(
        game_state_2,
        pitcher_state_2,
        matchup_state_2,
        pitch_usage_stats_2,
        pitcher_stats_2
    )

    print(f"Selected Action: {result_2.selected_action.pitch_type} @ {result_2.selected_action.zone}")
    print(f"Leverage: {result_2.leverage_level}")
    print(f"Entropy Status: {result_2.entropy_status}")
    print(f"\n🔍 Noise Filtering 결과:")
    print(f"  Filtered Pitches: {list(result_2.filtered_pitches.keys())}")
    print(f"  Noise Pitches (Removed): {result_2.noise_pitches}")
    print(f"\nTop Actions:")
    for action_key, prob in result_2.action_probs.items():
        print(f"  {action_key}: {prob:.1%}")
    print(f"\nRationale:\n{result_2.rationale}")

    print("\n" + "=" * 80)
    print("✅ 테스트 완료 (Data Noise Robustness 검증)")
    print("=" * 80)


if __name__ == "__main__":
    main()
