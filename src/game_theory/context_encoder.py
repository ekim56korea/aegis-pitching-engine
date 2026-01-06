"""
ContextEncoder: 게임 상태를 Neural Network 입력용 벡터로 변환

게임 상황(Count, Runners, TTO 등)을 One-Hot/Continuous Features로 인코딩하여
강화학습 에이전트나 예측 모델에 입력 가능한 형태로 변환

Key Features:
- Count One-Hot Encoding (12 dims)
- Runners One-Hot Encoding (8 dims)
- Outs One-Hot Encoding (3 dims)
- Times Through Order (TTO) One-Hot (4 dims) - 타자 순환 효과
- Pitcher Role One-Hot (2 dims) - SP/RP 분류 (NEW)
- Platoon Matchup Binary (1 dim) - Same-handed vs Opposite
- Batter Threat Matrix (5 dims) - 타자 프로필 입체화:
  * Chase Rate (유인구 스윙 비율)
  * Whiff Rate (헛스윙률 - 삼진 가능성)
  * ISO (Isolated Power - 장타력)
  * GB/FB Ratio (땅볼/뜬공 비율 - 병살 유도)
  * OPS (종합 타격 능력)
- Fatigue Index (1 dim) - SP/RP 별 상대적 피로도 (NEW)
- Game Context: Entropy, Score Diff, Inning, Prev Velo

Reference:
- Marchi & Albert (2016). "Analyzing Baseball Data with R"
- TTO Effect: 타자가 투수를 여러 번 볼수록 유리 (1회전 < 2회전 < 3회전)
- Platoon Advantage: 반대 타석 유리 (RHP vs LHB, LHP vs RHB)
- Leverage Index: 이닝, 점수차, 주자/아웃 조합으로 승부처 학습
- Sabermetrics: ISO, wOBA, O-Swing% 등 고급 지표
"""

from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn


class ContextEncoder:
    """
    게임 상태를 PyTorch Tensor로 인코딩

    Input Structure:
        game_state: {
            'outs': 2,               # 아웃 카운트 (0, 1, 2) - NEW
            'count': '1-2',          # 볼-스트라이크
            'runners': [1, 0, 0],    # [1루, 2루, 3루] (0=없음, 1=있음)
            'score_diff': 2,         # 우리팀 - 상대팀
            'inning': 5              # 현재 이닝
        }

        pitcher_state: {
            'hand': 'R',             # 투수 타석 ('L' or 'R')
            'role': 'SP',            # 투수 보직 ('SP' or 'RP') - NEW
            'pitch_count': 85,       # 누적 투구 수 (SP: 0~120, RP: 0~40)
            'entropy': 0.85,         # 투구 무작위성
            'prev_pitch': 'FF',      # 이전 구종
            'prev_velo': 98.2        # 이전 구속
        }

        matchup_state: {
            'batter_hand': 'L',      # 타자 타석 ('L' or 'R')
            'times_faced': 2,        # TTO: 0=첫 대면, 1=2번째, 2=3번째, 3+=4번째 이상
            # Batter Threat Matrix (타자 프로필)
            'chase_rate': 0.32,      # O-Swing%: 유인구 스윙 비율 (0.0~1.0)
            'whiff_rate': 0.28,      # Whiff%: 헛스윙률 (0.0~0.5), 높으면 삼진 쉬움
            'iso': 0.180,            # Isolated Power: 장타력 (0.0~0.4), 0.3+ 매우 위험
            'gb_fb_ratio': 1.2,      # GB/FB: 땅볼/뜬공 비율 (0.5~2.5), 높으면 병살 유도 가능
            'ops': 0.750             # OPS: 종합 타격 수준 (0.500~1.100)
        }

    Output:
        torch.FloatTensor with shape [1, total_dim]

    Feature Dimensions:
        - Count One-Hot: 12 (0-0 ~ 3-2)
        - Runners One-Hot: 8 (000 ~ 111)
        - Outs One-Hot: 3 (0, 1, 2)
        - TTO One-Hot: 4 (1st, 2nd, 3rd, 4th+)
        - Batter Hand One-Hot: 2 (L, R)
        - Pitcher Role One-Hot: 2 (SP, RP) - NEW
        - Platoon Matchup: 1 (binary: 1=same, 0=opposite)
        - Game Context: 4 (entropy, score_diff, inning, prev_velo)
        - Fatigue Index: 1 (SP: pitch_count/100, RP: pitch_count/30) - NEW
        - Batter Threat Matrix: 5 (chase_rate, whiff_rate, iso, gb_fb_ratio, ops)
        - Total: 12 + 8 + 3 + 4 + 2 + 2 + 1 + 4 + 1 + 5 = 42 dims
    """

    # Count 조합 (볼-스트라이크)
    COUNT_STATES = [
        '0-0', '0-1', '0-2',
        '1-0', '1-1', '1-2',
        '2-0', '2-1', '2-2',
        '3-0', '3-1', '3-2'
    ]

    # Runners 조합 (8가지)
    RUNNER_STATES = [
        (0, 0, 0),  # 주자 없음
        (1, 0, 0),  # 1루
        (0, 1, 0),  # 2루
        (0, 0, 1),  # 3루
        (1, 1, 0),  # 1,2루
        (1, 0, 1),  # 1,3루
        (0, 1, 1),  # 2,3루
        (1, 1, 1)   # 만루
    ]

    # TTO (Times Through Order) - 핵심 MLB 지표
    TTO_LEVELS = 4  # 1st, 2nd, 3rd, 4th+

    # Outs (아웃 카운트)
    OUT_STATES = 3  # 0, 1, 2

    def __init__(self, device: str = 'cpu'):
        """
        ContextEncoder 초기화

        Args:
            device: 'cpu' or 'cuda'
        """
        self.device = torch.device(device)

        # Feature dimensions
        self.count_dim = len(self.COUNT_STATES)      # 12
        self.runners_dim = len(self.RUNNER_STATES)   # 8
        self.outs_dim = self.OUT_STATES              # 3
        self.tto_dim = self.TTO_LEVELS               # 4
        self.hand_dim = 2                            # L, R
        self.role_dim = 2                            # SP, RP (NEW)
        self.platoon_dim = 1                         # Same/Opposite
        self.continuous_dim = 10                     # game_context(4) + fatigue(1) + batter_threat(5)

        self.total_dim = (
            self.count_dim +
            self.runners_dim +
            self.outs_dim +
            self.tto_dim +
            self.hand_dim +
            self.role_dim +
            self.platoon_dim +
            self.continuous_dim
        )

        print(f"✅ ContextEncoder 초기화")
        print(f"   Device: {self.device}")
        print(f"   Total Input Dim: {self.total_dim}")
        print(f"     - Count One-Hot: {self.count_dim}")
        print(f"     - Runners One-Hot: {self.runners_dim}")
        print(f"     - Outs One-Hot: {self.outs_dim}")
        print(f"     - TTO One-Hot: {self.tto_dim}")
        print(f"     - Batter Hand One-Hot: {self.hand_dim}")
        print(f"     - Pitcher Role One-Hot: {self.role_dim}")
        print(f"     - Platoon Matchup: {self.platoon_dim}")
        print(f"     - Continuous: {self.continuous_dim}")
        print(f"       * Game Context: 4")
        print(f"       * Fatigue Index: 1")
        print(f"       * Batter Threat Matrix: 5")

    def encode(
        self,
        game_state: Dict,
        pitcher_state: Dict,
        matchup_state: Dict
    ) -> torch.FloatTensor:
        """
        게임 상태를 벡터로 인코딩

        Args:
            game_state: {'outs', 'count', 'runners', 'score_diff', 'inning'}
            pitcher_state: {
                'hand', 'role', 'pitch_count',
                'entropy', 'prev_pitch', 'prev_velo'
            }
            matchup_state: {
                'batter_hand', 'times_faced',
                'chase_rate', 'whiff_rate', 'iso', 'gb_fb_ratio', 'ops'
            }

        Returns:
            encoded_tensor: torch.FloatTensor with shape [1, total_dim]
        """
        features = []

        # 1. Count One-Hot Encoding (12 dims)
        count_onehot = self._encode_count(game_state['count'])
        features.append(count_onehot)

        # 2. Runners One-Hot Encoding (8 dims)
        runners_onehot = self._encode_runners(game_state['runners'])
        features.append(runners_onehot)

        # 3. Outs One-Hot Encoding (3 dims) - NEW
        outs_onehot = self._encode_outs(game_state['outs'])
        features.append(outs_onehot)

        # 4. TTO (Times Through Order) One-Hot (4 dims) - 핵심!
        tto_onehot = self._encode_tto(matchup_state['times_faced'])
        features.append(tto_onehot)

        # 5. Batter Hand One-Hot (2 dims)
        hand_onehot = self._encode_batter_hand(matchup_state['batter_hand'])
        features.append(hand_onehot)

        # 6. Pitcher Role One-Hot (2 dims) - NEW
        role_onehot = self._encode_pitcher_role(pitcher_state['role'])
        features.append(role_onehot)

        # 7. Platoon Matchup Binary (1 dim)
        platoon_binary = self._encode_platoon_matchup(
            pitcher_state['hand'],
            matchup_state['batter_hand']
        )
        features.append(platoon_binary)

        # 8. Continuous Features (10 dims)
        # === Game Context (4 dims) ===
        # 8-1. Entropy (0 ~ 1, 이미 정규화됨)
        entropy = torch.tensor([pitcher_state['entropy']], dtype=torch.float32)
        features.append(entropy)

        # 8-2. Score Diff (Clipping + Normalization)
        score_diff_norm = self._normalize_score_diff(game_state['score_diff'])
        features.append(score_diff_norm)

        # 8-3. Inning (Normalization: 1~9 -> 0~1)
        inning_norm = torch.tensor(
            [(game_state['inning'] - 1) / 8.0],  # 1~9 -> 0~1
            dtype=torch.float32
        )
        features.append(inning_norm)

        # 8-4. Previous Velocity (Normalization: 70~105 mph -> 0~1)
        prev_velo = pitcher_state.get('prev_velo', 90.0)  # default 90 mph
        velo_norm = torch.tensor(
            [(prev_velo - 70.0) / 35.0],  # 70~105 -> 0~1
            dtype=torch.float32
        )
        features.append(velo_norm)

        # === Fatigue Index (1 dim) - NEW ===
        # 8-5. Fatigue Index (SP/RP별 상대적 피로도)
        fatigue_index = self._calculate_fatigue(
            pitcher_state.get('pitch_count', 0),
            pitcher_state.get('role', 'SP')
        )
        features.append(fatigue_index)

        # === Batter Threat Matrix (5 dims) ===
        # 7-6. Chase Rate (O-Swing%: 유인구 스윙 비율)
        chase_rate = matchup_state.get('chase_rate', 0.3)  # default 30%
        chase_rate_tensor = torch.tensor([chase_rate], dtype=torch.float32)
        features.append(chase_rate_tensor)

        # 7-7. Whiff Rate (헛스윙률: 0.0 ~ 0.5)
        whiff_rate = matchup_state.get('whiff_rate', 0.25)  # default 25%
        whiff_rate_norm = torch.tensor(
            [whiff_rate / 0.5],  # 0.0~0.5 -> 0~1
            dtype=torch.float32
        )
        features.append(whiff_rate_norm)

        # 7-8. ISO (Isolated Power: 0.0 ~ 0.4)
        iso = matchup_state.get('iso', 0.150)  # default .150
        iso_norm = torch.tensor(
            [iso / 0.4],  # 0.0~0.4 -> 0~1, 0.3+ (0.75+) = 매우 위험
            dtype=torch.float32
        )
        features.append(iso_norm)

        # 7-9. GB/FB Ratio (땅볼/뜬공 비율: 0.5 ~ 2.5)
        gb_fb_ratio = matchup_state.get('gb_fb_ratio', 1.0)  # default 1.0 (균형)
        gb_fb_norm = torch.tensor(
            [(gb_fb_ratio - 0.5) / 2.0],  # 0.5~2.5 -> 0~1
            dtype=torch.float32
        )
        features.append(gb_fb_norm)

        # 7-10. OPS (종합 타격 수준: 0.500 ~ 1.100)
        ops = matchup_state.get('ops', 0.700)  # default .700 (평균)
        ops_norm = torch.tensor(
            [(ops - 0.500) / 0.600],  # 0.500~1.100 -> 0~1
            dtype=torch.float32
        )
        features.append(ops_norm)

        # Concatenate all features
        encoded = torch.cat(features, dim=0)

        # Add batch dimension: [total_dim] -> [1, total_dim]
        encoded = encoded.unsqueeze(0)

        return encoded.to(self.device)

    def _encode_count(self, count: str) -> torch.FloatTensor:
        """
        Count를 One-Hot Encoding

        Args:
            count: '1-2' (볼-스트라이크)

        Returns:
            one_hot: [12] (0-0 ~ 3-2)
        """
        if count not in self.COUNT_STATES:
            raise ValueError(f"Invalid count: {count}. Must be one of {self.COUNT_STATES}")

        one_hot = torch.zeros(self.count_dim, dtype=torch.float32)
        idx = self.COUNT_STATES.index(count)
        one_hot[idx] = 1.0

        return one_hot

    def _encode_runners(self, runners: List[int]) -> torch.FloatTensor:
        """
        Runners를 One-Hot Encoding

        Args:
            runners: [1, 0, 0] (1루, 2루, 3루)

        Returns:
            one_hot: [8] (000 ~ 111)
        """
        if len(runners) != 3:
            raise ValueError(f"Runners must have 3 elements, got: {len(runners)}")

        runners_tuple = tuple(runners)
        if runners_tuple not in self.RUNNER_STATES:
            raise ValueError(f"Invalid runners: {runners}. Must be binary [0 or 1, 0 or 1, 0 or 1]")

        one_hot = torch.zeros(self.runners_dim, dtype=torch.float32)
        idx = self.RUNNER_STATES.index(runners_tuple)
        one_hot[idx] = 1.0

        return one_hot

    def _encode_outs(self, outs: int) -> torch.FloatTensor:
        """
        아웃 카운트를 One-Hot Encoding

        Args:
            outs: 0, 1, or 2

        Returns:
            one_hot: [3] (0, 1, 2)

        Note:
            아웃 카운트는 게임 상황의 핵심 요소:
            - 0 아웃: 득점 기회 많음
            - 1 아웃: 균형
            - 2 아웃: 압박 상황 (투수/타자 모두)
        """
        if outs not in [0, 1, 2]:
            raise ValueError(f"outs must be 0, 1, or 2, got: {outs}")

        one_hot = torch.zeros(self.outs_dim, dtype=torch.float32)
        one_hot[outs] = 1.0

        return one_hot

    def _encode_tto(self, times_faced: int) -> torch.FloatTensor:
        """
        TTO (Times Through Order)를 One-Hot Encoding

        Args:
            times_faced: 0 (첫 대면), 1 (두번째), 2 (세번째), 3+ (네번째 이상)

        Returns:
            one_hot: [4] (1st, 2nd, 3rd, 4th+)

        Note:
            TTO Effect는 MLB에서 중요한 지표:
            - 1회전: 타자가 처음 보는 투수 (투수 유리)
            - 2회전: 타자가 적응 시작 (균형)
            - 3회전 이상: 타자가 완전히 적응 (타자 유리)
        """
        if times_faced < 0:
            raise ValueError(f"times_faced must be >= 0, got: {times_faced}")

        # Clipping: 3+ 이상은 모두 동일하게 취급
        tto_level = min(times_faced, self.TTO_LEVELS - 1)

        one_hot = torch.zeros(self.tto_dim, dtype=torch.float32)
        one_hot[tto_level] = 1.0

        return one_hot

    def _encode_batter_hand(self, batter_hand: str) -> torch.FloatTensor:
        """
        타자 타석을 One-Hot Encoding

        Args:
            batter_hand: 'L' or 'R'

        Returns:
            one_hot: [2] (L, R)
        """
        if batter_hand not in ['L', 'R']:
            raise ValueError(f"batter_hand must be 'L' or 'R', got: {batter_hand}")

        one_hot = torch.zeros(self.hand_dim, dtype=torch.float32)
        if batter_hand == 'L':
            one_hot[0] = 1.0
        else:  # 'R'
            one_hot[1] = 1.0

        return one_hot

    def _encode_pitcher_role(self, role: str) -> torch.FloatTensor:
        """
        Pitcher role을 One-Hot Encoding

        Args:
            role: 'SP' (Starter) or 'RP' (Reliever)

        Returns:
            one_hot: [2] (SP, RP)

        Note:
            선발과 불펜은 완전히 다른 사용 패턴:
            - SP: 5-7 이닝, 80-110 투구, TTO 중요
            - RP: 1-2 이닝, 15-30 투구, 단기 전력 투입
        """
        if role not in ['SP', 'RP']:
            raise ValueError(f"role must be 'SP' or 'RP', got: {role}")

        one_hot = torch.zeros(self.role_dim, dtype=torch.float32)
        if role == 'SP':
            one_hot[0] = 1.0
        else:  # 'RP'
            one_hot[1] = 1.0

        return one_hot

    def _encode_platoon_matchup(
        self,
        pitcher_hand: str,
        batter_hand: str
    ) -> torch.FloatTensor:
        """
        Platoon Matchup을 Binary Encoding

        Args:
            pitcher_hand: 'L' or 'R'
            batter_hand: 'L' or 'R'

        Returns:
            binary: [1] (1.0 = Same-handed, 0.0 = Opposite)

        Note:
            Platoon Advantage (반대 타석 유리):
            - RHP vs LHB: 타자 유리 (Opposite)
            - LHP vs RHB: 타자 유리 (Opposite)
            - RHP vs RHB: 투수 유리 (Same)
            - LHP vs LHB: 투수 유리 (Same)

            MLB 통계상 Opposite matchup에서 타자 OPS가 약 50-100점 높음
        """
        if pitcher_hand not in ['L', 'R']:
            raise ValueError(f"pitcher_hand must be 'L' or 'R', got: {pitcher_hand}")
        if batter_hand not in ['L', 'R']:
            raise ValueError(f"batter_hand must be 'L' or 'R', got: {batter_hand}")

        # Same-handed = 1.0, Opposite = 0.0
        is_same_handed = 1.0 if pitcher_hand == batter_hand else 0.0

        return torch.tensor([is_same_handed], dtype=torch.float32)

    def _normalize_score_diff(self, score_diff: int) -> torch.FloatTensor:
        """
        점수 차이를 정규화

        Args:
            score_diff: 우리팀 - 상대팀 (예: +3 = 3점 앞섬, -2 = 2점 뒤짐)

        Returns:
            normalized: [1] (-5 ~ +5 clipping, then -1 ~ +1 normalization)
        """
        # Clipping: -5 ~ +5
        clipped = max(-5, min(5, score_diff))

        # Normalization: -5 ~ +5 -> -1 ~ +1
        normalized = clipped / 5.0

        return torch.tensor([normalized], dtype=torch.float32)

    def _calculate_fatigue(self, pitch_count: int, role: str) -> torch.FloatTensor:
        """
        Calculate relative fatigue index based on pitcher role

        Args:
            pitch_count: Current pitch count (0~120 for SP, 0~40 for RP)
            role: 'SP' (Starter) or 'RP' (Reliever)

        Returns:
            fatigue_index: [1] (can exceed 1.0 for extreme cases)

        Logic:
            - SP: pitch_count / 100.0 (100 pitches = baseline, 110-120 = overwork)
            - RP: pitch_count / 30.0 (30 pitches = baseline, 35+ = rapid fatigue)

        Note:
            선발과 불펜의 체력 관리는 근본적으로 다름:
            - SP: 서서히 지치며 90-100개 이후 급격한 피로
            - RP: 짧고 강하게, 30개 이후 급격한 성능 저하
        """
        if role not in ['SP', 'RP']:
            raise ValueError(f"role must be 'SP' or 'RP', got: {role}")

        if role == 'SP':
            fatigue = pitch_count / 100.0  # SP baseline: 100 pitches
        else:  # 'RP'
            fatigue = pitch_count / 30.0   # RP baseline: 30 pitches

        return torch.tensor([fatigue], dtype=torch.float32)

    def get_input_dim(self) -> int:
        """
        총 입력 차원 수 반환

        Returns:
            total_dim: int (예: 30)
        """
        return self.total_dim

    def decode_count(self, count_onehot: torch.Tensor) -> str:
        """
        Count One-Hot을 문자열로 역변환 (디버깅용)

        Args:
            count_onehot: [12] one-hot tensor

        Returns:
            count: '1-2'
        """
        idx = torch.argmax(count_onehot).item()
        return self.COUNT_STATES[idx]

    def decode_runners(self, runners_onehot: torch.Tensor) -> Tuple[int, int, int]:
        """
        Runners One-Hot을 튜플로 역변환 (디버깅용)

        Args:
            runners_onehot: [8] one-hot tensor

        Returns:
            runners: (1, 0, 0)
        """
        idx = torch.argmax(runners_onehot).item()
        return self.RUNNER_STATES[idx]

    def decode_tto(self, tto_onehot: torch.Tensor) -> int:
        """
        TTO One-Hot을 정수로 역변환 (디버깅용)

        Args:
            tto_onehot: [4] one-hot tensor

        Returns:
            times_faced: 0, 1, 2, or 3+
        """
        idx = torch.argmax(tto_onehot).item()
        return idx


def main():
    """사용 예시 및 테스트"""
    print("=" * 80)
    print("🎯 ContextEncoder 테스트")
    print("=" * 80 + "\n")

    # Encoder 초기화
    encoder = ContextEncoder(device='cpu')
    print()

    # ========================================================================
    # Test Case 1: 일반적인 상황 (좋은 선구안, 컨택 타자)
    # ========================================================================
    print("📊 Test Case 1: Contact Hitter with Good Discipline")
    print("-" * 80)

    game_state_1 = {
        'outs': 1,             # 1 아웃
        'count': '1-2',
        'runners': [1, 0, 0],  # 1루 주자
        'score_diff': 2,       # 2점 리드
        'inning': 5            # 5회
    }

    pitcher_state_1 = {
        'hand': 'R',           # 우투수
        'role': 'SP',          # 선발 투수
        'pitch_count': 65,     # 65구
        'entropy': 0.85,
        'prev_pitch': 'FF',
        'prev_velo': 98.2
    }

    matchup_state_1 = {
        'batter_hand': 'L',    # 좌타자 (Opposite = 타자 유리)
        'times_faced': 1,      # 2번째 대면 (2회전)
        # Batter Threat Matrix
        'chase_rate': 0.25,    # 25% - 좋은 선구안 (낮음 = 유인구에 잘 안 속음)
        'whiff_rate': 0.18,    # 18% - 낮음 (컨택 잘 함, 삼진 어려움)
        'iso': 0.140,          # .140 - 낮은 파워 (장타 위협 적음)
        'gb_fb_ratio': 1.8,    # 1.8 - 땅볼 타자 (병살타 유도 가능)
        'ops': 0.720           # .720 - 평균 수준
    }

    encoded_1 = encoder.encode(game_state_1, pitcher_state_1, matchup_state_1)

    print(f"Input:")
    print(f"  Outs: {game_state_1['outs']}")
    print(f"  Count: {game_state_1['count']}")
    print(f"  Runners: {game_state_1['runners']}")
    print(f"  Pitcher: {pitcher_state_1['hand']}HP, {pitcher_state_1['pitch_count']} pitches")
    print(f"  Batter: {matchup_state_1['batter_hand']}HB (Opposite Matchup)")
    print(f"\n  Batter Profile:")
    print(f"    Chase Rate: {matchup_state_1['chase_rate']:.1%} (좋은 선구안)")
    print(f"    Whiff Rate: {matchup_state_1['whiff_rate']:.1%} (컨택 좋음)")
    print(f"    ISO: {matchup_state_1['iso']:.3f} (낮은 파워)")
    print(f"    GB/FB: {matchup_state_1['gb_fb_ratio']:.2f} (땅볼 타자)")
    print(f"    OPS: {matchup_state_1['ops']:.3f} (평균)")
    print(f"  Strategy: 병살타 유도, 존 공략보다 약점 공략")

    print(f"\nOutput:")
    print(f"  Encoded Shape: {encoded_1.shape}")
    print(f"  Encoded Tensor (first 10 dims): {encoded_1[0, :10].tolist()}")
    print(f"  Total Dims: {encoder.get_input_dim()}")

    # ========================================================================
    # Test Case 2: 위기 상황 (파워 히터, 나쁜 선구안)
    # ========================================================================
    print("\n" + "=" * 80)
    print("📊 Test Case 2: Power Hitter with Poor Discipline (위기)")
    print("-" * 80)

    game_state_2 = {
        'outs': 2,             # 2 아웃 (압박)
        'count': '3-2',
        'runners': [1, 1, 1],  # 만루
        'score_diff': -1,      # 1점 뒤짐
        'inning': 9            # 9회
    }

    pitcher_state_2 = {
        'hand': 'L',           # 좌투수
        'role': 'RP',          # 불펜 투수 (9회 마무리 상황)
        'pitch_count': 35,     # 35구 (RP로서는 많은 편)
        'entropy': 0.45,       # 낮은 엔트로피 (예측 가능)
        'prev_pitch': 'SL',
        'prev_velo': 85.3
    }

    matchup_state_2 = {
        'batter_hand': 'R',    # 우타자 (Opposite = 타자 유리)
        'times_faced': 2,      # 3번째 대면 (3회전, 타자 유리)
        # Batter Threat Matrix - 위험한 파워 히터
        'chase_rate': 0.42,    # 42% - 나쁜 선구안 (높음 = 유인구 전략 유효)
        'whiff_rate': 0.32,    # 32% - 높음 (삼진 가능)
        'iso': 0.280,          # .280 - 높은 파워 (매우 위험! 장타 주의)
        'gb_fb_ratio': 0.7,    # 0.7 - 플라이볼 타자 (홈런 위험)
        'ops': 0.880           # .880 - 높은 수준 (위협적)
    }

    encoded_2 = encoder.encode(game_state_2, pitcher_state_2, matchup_state_2)

    print(f"Input:")
    print(f"  Outs: {game_state_2['outs']} (2 outs!)")
    print(f"  Count: {game_state_2['count']} (Full Count)")
    print(f"  Runners: {game_state_2['runners']} (만루)")
    print(f"  Score Diff: {game_state_2['score_diff']} (1점 뒤짐)")
    print(f"  Inning: {game_state_2['inning']} (9회)")
    print(f"  Pitcher: {pitcher_state_2['hand']}HP ({pitcher_state_2['role']}), {pitcher_state_2['pitch_count']} pitches")
    print(f"  ⚠️ Reliever at 35 pitches = High fatigue (35/30 = 1.17)")
    print(f"  Batter: {matchup_state_2['batter_hand']}HB (Opposite - 타자 유리)")
    print(f"\n  Batter Profile:")
    print(f"    Chase Rate: {matchup_state_2['chase_rate']:.1%} (나쁜 선구안)")
    print(f"    Whiff Rate: {matchup_state_2['whiff_rate']:.1%} (삼진 가능)")
    print(f"    ISO: {matchup_state_2['iso']:.3f} (⚠️ 위험한 파워!)")
    print(f"    GB/FB: {matchup_state_2['gb_fb_ratio']:.2f} (플라이볼 타자)")
    print(f"    OPS: {matchup_state_2['ops']:.3f} (높은 수준)")
    print(f"  TTO: {matchup_state_2['times_faced']} (3rd time)")
    print(f"  Strategy: 유인구로 삼진, 존 안쪽 공 금지 (장타 주의)")

    print(f"\nOutput:")
    print(f"  Encoded Shape: {encoded_2.shape}")
    print(f"  Total Non-Zero Features: {(encoded_2 != 0).sum().item()}")

    # ========================================================================
    # Test Case 3: 평균적인 타자 (균형잡힌 프로필)
    # ========================================================================
    print("\n" + "=" * 80)
    print("📊 Test Case 3: Balanced Batter (평균 수준)")
    print("-" * 80)

    game_state_3 = {
        'outs': 0,             # 0 아웃
        'count': '0-0',
        'runners': [0, 0, 0],  # 주자 없음
        'score_diff': 5,       # 5점 리드 (clipping 테스트)
        'inning': 1
    }

    pitcher_state_3 = {
        'role': 'SP',          # 선발 투수
        'hand': 'R',           # 우투수
        'pitch_count': 12,     # 초반
        'entropy': 0.92,       # 높은 엔트로피 (예측 불가)
        'prev_pitch': None,    # 첫 투구
        'prev_velo': 90.0      # default
    }

    matchup_state_3 = {
        'batter_hand': 'R',    # 우타자 (Same = 투수 유리)
        'times_faced': 0,      # 첫 대면 (1회전, 투수 유리)
        # Batter Threat Matrix - 평균적인 타자
        'chase_rate': 0.31,    # 31% - 평균 선구안
        'whiff_rate': 0.25,    # 25% - 평균 컨택
        'iso': 0.155,          # .155 - 평균 파워
        'gb_fb_ratio': 1.0,    # 1.0 - 균형잡힌 타구 (땅볼/뜬공 비슷)
        'ops': 0.710           # .710 - 평균 수준
    }

    encoded_3 = encoder.encode(game_state_3, pitcher_state_3, matchup_state_3)

    print(f"Input:")
    print(f"  Outs: {game_state_3['outs']}")
    print(f"  Count: {game_state_3['count']} (초구)")
    print(f"  Runners: {game_state_3['runners']} (주자 없음)")
    print(f"  Score Diff: {game_state_3['score_diff']} (5점 리드, clipped)")
    print(f"  Inning: {game_state_3['inning']}")
    print(f"  Pitcher: {pitcher_state_3['hand']}HP, {pitcher_state_3['pitch_count']} pitches")
    print(f"  Batter: {matchup_state_3['batter_hand']}HB (Same Matchup)")
    print(f"\n  Batter Profile:")
    print(f"    Chase Rate: {matchup_state_3['chase_rate']:.1%} (평균)")
    print(f"    Whiff Rate: {matchup_state_3['whiff_rate']:.1%} (평균)")
    print(f"    ISO: {matchup_state_3['iso']:.3f} (평균)")
    print(f"    GB/FB: {matchup_state_3['gb_fb_ratio']:.2f} (균형)")
    print(f"    OPS: {matchup_state_3['ops']:.3f} (평균)")
    print(f"  TTO: {matchup_state_3['times_faced']} (1st time, 투수 유리)")
    print(f"  Strategy: 표준 배합, 다양한 전략 가능")

    print(f"\nOutput:")
    print(f"  Encoded Shape: {encoded_3.shape}")

    # ========================================================================
    # Feature Breakdown (Test Case 1 기준)
    # ========================================================================
    print("\n" + "=" * 80)
    print("🔍 Feature Breakdown (Test Case 1)")
    print("-" * 80)

    vec = encoded_1[0]

    # Count (12 dims)
    count_vec = vec[:12]
    print(f"Count One-Hot ({encoder.decode_count(count_vec)}):")
    print(f"  {count_vec.tolist()}")

    # Runners (8 dims)
    runners_vec = vec[12:20]
    print(f"\nRunners One-Hot ({encoder.decode_runners(runners_vec)}):")
    print(f"  {runners_vec.tolist()}")

    # Outs (3 dims) - NEW
    outs_vec = vec[20:23]
    print(f"\nOuts One-Hot ({torch.argmax(outs_vec).item()} outs):")
    print(f"  {outs_vec.tolist()}")
    print(f"  [0, 1, 2]")

    # TTO (4 dims)
    tto_vec = vec[23:27]
    print(f"\nTTO One-Hot (times_faced={encoder.decode_tto(tto_vec)}):")
    print(f"  {tto_vec.tolist()}")
    print(f"  [1st, 2nd, 3rd, 4th+]")

    # Batter Hand (2 dims)
    hand_vec = vec[27:29]
    print(f"\nBatter Hand One-Hot:")
    print(f"  {hand_vec.tolist()}")
    print(f"  [L, R]")

    # Pitcher Role (2 dims) - NEW
    role_vec = vec[29:31]
    role_name = "SP" if role_vec[0].item() == 1.0 else "RP"
    print(f"\nPitcher Role One-Hot ({role_name}):")
    print(f"  {role_vec.tolist()}")
    print(f"  [SP, RP]")

    # Platoon Matchup (1 dim)
    platoon_vec = vec[31]
    matchup_type = "Same-handed" if platoon_vec.item() == 1.0 else "Opposite"
    print(f"\nPlatoon Matchup Binary:")
    print(f"  {platoon_vec.item():.1f} ({matchup_type})")
    print(f"  1.0 = Same-handed (투수 유리), 0.0 = Opposite (타자 유리)")

    # Continuous (10 dims)
    continuous_vec = vec[32:]
    print(f"\nContinuous Features (Game Context + Fatigue + Batter Threat):")
    print(f"  Game Context (4 dims):")
    print(f"    Entropy: {continuous_vec[0]:.4f}")
    print(f"    Score Diff (norm): {continuous_vec[1]:.4f}")
    print(f"    Inning (norm): {continuous_vec[2]:.4f}")
    print(f"    Prev Velo (norm): {continuous_vec[3]:.4f}")
    print(f"  Fatigue Index (1 dim):")
    print(f"    Fatigue: {continuous_vec[4]:.4f} (SP: {pitcher_state_1['pitch_count']}/100)")
    print(f"  Batter Threat Matrix (5 dims):")
    print(f"    Chase Rate: {continuous_vec[5]:.4f} ({matchup_state_1['chase_rate']:.1%})")
    print(f"    Whiff Rate (norm): {continuous_vec[6]:.4f} ({matchup_state_1['whiff_rate']:.1%} / 0.5)")
    print(f"    ISO (norm): {continuous_vec[7]:.4f} ({matchup_state_1['iso']:.3f} / 0.4)")
    print(f"    GB/FB (norm): {continuous_vec[8]:.4f} (({matchup_state_1['gb_fb_ratio']:.2f} - 0.5) / 2.0)")
    print(f"    OPS (norm): {continuous_vec[9]:.4f} (({matchup_state_1['ops']:.3f} - 0.5) / 0.6)")

    # ========================================================================
    # Batter Threat Matrix 비교
    # ========================================================================
    print("\n" + "=" * 80)
    print("🎯 Batter Threat Matrix 비교")
    print("=" * 80 + "\n")

    vec2 = encoded_2[0]
    vec3 = encoded_3[0]

    print("┌─────────────┬──────────────┬──────────────┬──────────────┐")
    print("│   Feature   │   Case 1     │   Case 2     │   Case 3     │")
    print("│             │  (Contact)   │   (Power)    │  (Average)   │")
    print("├─────────────┼──────────────┼──────────────┼──────────────┤")
    print(f"│ Chase Rate  │    {matchup_state_1['chase_rate']:.1%}     │    {matchup_state_2['chase_rate']:.1%}     │    {matchup_state_3['chase_rate']:.1%}     │")
    print(f"│ Whiff Rate  │    {matchup_state_1['whiff_rate']:.1%}     │    {matchup_state_2['whiff_rate']:.1%}     │    {matchup_state_3['whiff_rate']:.1%}     │")
    print(f"│ ISO         │    {matchup_state_1['iso']:.3f}    │    {matchup_state_2['iso']:.3f}    │    {matchup_state_3['iso']:.3f}    │")
    print(f"│ GB/FB       │    {matchup_state_1['gb_fb_ratio']:.2f}      │    {matchup_state_2['gb_fb_ratio']:.2f}      │    {matchup_state_3['gb_fb_ratio']:.2f}      │")
    print(f"│ OPS         │    {matchup_state_1['ops']:.3f}    │    {matchup_state_2['ops']:.3f}    │    {matchup_state_3['ops']:.3f}    │")
    print("└─────────────┴──────────────┴──────────────┴──────────────┘")

    print("\nThreat Level Analysis:")
    print(f"  Case 1: 낮은 위협도 (땅볼 타자, 낮은 파워)")
    print(f"  Case 2: ⚠️ 높은 위협도 (파워 히터, 플라이볼, 유인구 전략)")
    print(f"  Case 3: 중간 위협도 (평균적, 표준 배합)")

    # ========================================================================
    # Platoon Matchup 비교
    # ========================================================================
    print("\n" + "=" * 80)
    print("🔄 Platoon Matchup 비교")
    print("=" * 80 + "\n")

    print("\nCase 1 (RHP vs LHB):")
    print(f"  Platoon: {vec[31].item():.1f} (Opposite = 타자 유리)")

    print("\nCase 2 (LHP vs RHB):")
    print(f"  Platoon: {vec2[31].item():.1f} (Opposite = 타자 유리)")

    print("\nCase 3 (RHP vs RHB):")
    print(f"  Platoon: {vec3[31].item():.1f} (Same = 투수 유리)")

    # ========================================================================
    # Fatigue Index 비교 (SP vs RP)
    # ========================================================================
    print("\n" + "=" * 80)
    print("⚡ Fatigue Index 비교 (SP vs RP)")
    print("=" * 80 + "\n")

    print(f"Case 1 (SP, {pitcher_state_1['pitch_count']} pitches):")
    print(f"  Fatigue Index: {vec[36]:.3f} (= {pitcher_state_1['pitch_count']} / 100)")
    print(f"  Status: 중반, 여유 있음")

    print(f"\nCase 2 (RP, {pitcher_state_2['pitch_count']} pitches):")
    print(f"  Fatigue Index: {vec2[36]:.3f} (= {pitcher_state_2['pitch_count']} / 30)")
    print(f"  Status: ⚠️ High fatigue! RP는 30구가 baseline (35구 = 과부하)")

    print(f"\nCase 3 (SP, {pitcher_state_3['pitch_count']} pitches):")
    print(f"  Fatigue Index: {vec3[36]:.3f} (= {pitcher_state_3['pitch_count']} / 100)")
    print(f"  Status: 초반, 최상의 컨디션")

    print("\n💡 SP vs RP Fatigue Model:")
    print("  - SP: 100구 기준 (80~100 정상, 110+ 과부하)")
    print("  - RP: 30구 기준 (20~30 정상, 35+ 급격한 성능 저하)")
    print("  - 같은 투구수라도 역할에 따라 피로도는 완전히 다름!")

    # ========================================================================
    # Batch Encoding Test
    # ========================================================================
    print("\n" + "=" * 80)
    print("🚀 Batch Encoding Test")
    print("-" * 80)

    # 여러 상태를 인코딩하여 Batch로 만들기
    states = [
        (game_state_1, pitcher_state_1, matchup_state_1),
        (game_state_2, pitcher_state_2, matchup_state_2),
        (game_state_3, pitcher_state_3, matchup_state_3)
    ]

    batch = []
    for gs, ps, ms in states:
        encoded = encoder.encode(gs, ps, ms)
        batch.append(encoded)

    # Stack to batch: [3, 1, total_dim] -> [3, total_dim]
    batch_tensor = torch.cat(batch, dim=0)

    print(f"Batch Shape: {batch_tensor.shape}")
    print(f"Expected: [3, {encoder.get_input_dim()}]")

    print("\n" + "=" * 80)
    print("✅ 완료")
    print("=" * 80)


if __name__ == "__main__":
    main()
