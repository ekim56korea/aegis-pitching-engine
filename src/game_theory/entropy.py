"""
EntropyMonitor: 투구 시퀀스의 무작위성(Randomness) 측정

Shannon Entropy를 사용하여 투수의 투구 패턴 예측 가능성 분석
- 높은 엔트로피: 타자가 예측하기 어려움 (Good)
- 낮은 엔트로피: 타자가 예측하기 쉬움 (Danger)

Reference: Shannon, C.E. (1948). "A Mathematical Theory of Communication"
"""

from collections import deque, Counter
from typing import Optional, Dict
import math


class EntropyMonitor:
    """
    투구 시퀀스의 무작위성을 Shannon Entropy로 측정

    Mathematical Foundation:
        H(S) = -Σ(p_i * log₂(p_i))

        where:
        - p_i: 최근 시퀀스 내에서 구종 i의 출현 확률
        - H(S): 0 (완전 예측 가능) ~ log₂(N) (완전 무작위)

    Usage:
        monitor = EntropyMonitor(window_size=20)
        monitor.update('FF')
        monitor.update('SL')
        entropy = monitor.calculate_entropy()
        status = monitor.get_predictability_status()
    """

    def __init__(self, window_size: int = 20):
        """
        EntropyMonitor 초기화

        Args:
            window_size: Sliding Window 크기 (최근 N개의 투구만 추적)
                        기본값 20 = 타석당 평균 투구 수 (4-5타석 분량)
        """
        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got: {window_size}")

        self.window_size = window_size
        self.pitch_queue: deque = deque(maxlen=window_size)

        print(f"✅ EntropyMonitor 초기화 (Window Size: {window_size})")

    def update(self, pitch_type: str) -> None:
        """
        새로운 투구를 기록하고 Sliding Window 업데이트

        Args:
            pitch_type: 구종 (예: 'FF', 'SL', 'CH', 'CU', etc.)

        Note:
            - deque의 maxlen 속성으로 자동으로 오래된 투구 제거
            - 큐가 가득 차면 가장 오래된 투구가 자동으로 pop됨
        """
        self.pitch_queue.append(pitch_type)

    def calculate_entropy(self, normalized: bool = True) -> float:
        """
        현재 Sliding Window의 Shannon Entropy 계산

        Args:
            normalized: True면 0~1 사이로 정규화 (log₂(N)으로 나눔)
                       False면 원본 엔트로피 값 (0 ~ log₂(N))

        Returns:
            entropy: 엔트로피 값
                    - 0.0: 완전히 예측 가능 (한 가지 구종만)
                    - 1.0 (normalized): 완전히 무작위 (모든 구종 균등)

        Formula:
            H(S) = -Σ(p_i * log₂(p_i))
            H_normalized = H(S) / log₂(N)

            where N = number of unique pitch types
        """
        # 데이터가 없으면 엔트로피 0
        if len(self.pitch_queue) == 0:
            return 0.0

        # 구종별 빈도수 계산
        pitch_counts = Counter(self.pitch_queue)
        total_pitches = len(self.pitch_queue)

        # Shannon Entropy 계산
        entropy = 0.0
        for count in pitch_counts.values():
            # 확률 p_i
            probability = count / total_pitches

            # -p_i * log₂(p_i)
            # log₂(x) = log(x) / log(2)
            entropy -= probability * math.log2(probability)

        # 정규화 (Optional)
        if normalized:
            # 최대 엔트로피: log₂(N), N = 고유 구종 개수
            num_unique_pitches = len(pitch_counts)

            if num_unique_pitches <= 1:
                # 한 가지 구종만 있으면 최대 엔트로피도 0
                return 0.0

            max_entropy = math.log2(num_unique_pitches)
            entropy = entropy / max_entropy

        return entropy

    def get_predictability_status(self) -> Dict[str, str]:
        """
        엔트로피 점수를 기반으로 예측 가능성 상태 반환

        Returns:
            status_dict: {
                'level': 'High' | 'Medium' | 'Low',
                'description': 상태 설명,
                'recommendation': 권장 사항
            }

        Thresholds:
            - High Entropy (> 0.8): Unpredictable (Good)
            - Medium Entropy (0.5 ~ 0.8): Moderate
            - Low Entropy (< 0.5): Predictable (Danger)
        """
        entropy = self.calculate_entropy(normalized=True)

        if entropy > 0.8:
            return {
                'level': 'High',
                'description': 'Unpredictable (Good)',
                'recommendation': '✅ 타자가 패턴을 읽기 매우 어려움'
            }
        elif entropy > 0.5:
            return {
                'level': 'Medium',
                'description': 'Moderate',
                'recommendation': '⚠️ 일부 패턴이 보일 수 있음 - 구종 믹스 개선 권장'
            }
        else:
            return {
                'level': 'Low',
                'description': 'Predictable (Danger)',
                'recommendation': '🚨 타자가 패턴을 쉽게 파악 - 즉시 전략 변경 필요'
            }

    def get_pitch_distribution(self) -> Dict[str, float]:
        """
        현재 Sliding Window의 구종별 분포 반환 (디버깅/분석용)

        Returns:
            distribution: {pitch_type: probability}
        """
        if len(self.pitch_queue) == 0:
            return {}

        pitch_counts = Counter(self.pitch_queue)
        total_pitches = len(self.pitch_queue)

        distribution = {
            pitch_type: count / total_pitches
            for pitch_type, count in pitch_counts.items()
        }

        return distribution

    def reset(self) -> None:
        """
        Sliding Window 초기화 (새로운 타자/이닝 시작 시)
        """
        self.pitch_queue.clear()
        print("🔄 EntropyMonitor 리셋")


def main():
    """사용 예시 및 테스트"""
    print("=" * 80)
    print("🎲 EntropyMonitor 테스트")
    print("=" * 80 + "\n")

    # ========================================================================
    # Case A: 직구만 10개 던짐 (Entropy 0 예상)
    # ========================================================================
    print("📊 Case A: 직구만 10개 (완전히 예측 가능)")
    print("-" * 80)

    monitor_a = EntropyMonitor(window_size=20)

    # 직구(FF)만 10개
    for i in range(10):
        monitor_a.update('FF')

    entropy_a = monitor_a.calculate_entropy(normalized=False)
    entropy_a_norm = monitor_a.calculate_entropy(normalized=True)
    status_a = monitor_a.get_predictability_status()
    dist_a = monitor_a.get_pitch_distribution()

    print(f"투구 수: {len(monitor_a.pitch_queue)}")
    print(f"구종 분포: {dist_a}")
    print(f"엔트로피 (원본): {entropy_a:.4f}")
    print(f"엔트로피 (정규화): {entropy_a_norm:.4f}")
    print(f"상태: {status_a['level']} - {status_a['description']}")
    print(f"권장: {status_a['recommendation']}\n")

    # ========================================================================
    # Case B: 직구, 커브를 번갈아 던짐 (중간 Entropy)
    # ========================================================================
    print("=" * 80)
    print("📊 Case B: 직구(FF), 커브(CU) 번갈아 던짐 (중간 예측 가능)")
    print("-" * 80)

    monitor_b = EntropyMonitor(window_size=20)

    # FF, CU 번갈아 20개
    for i in range(20):
        if i % 2 == 0:
            monitor_b.update('FF')
        else:
            monitor_b.update('CU')

    entropy_b = monitor_b.calculate_entropy(normalized=False)
    entropy_b_norm = monitor_b.calculate_entropy(normalized=True)
    status_b = monitor_b.get_predictability_status()
    dist_b = monitor_b.get_pitch_distribution()

    print(f"투구 수: {len(monitor_b.pitch_queue)}")
    print(f"구종 분포: {dist_b}")
    print(f"엔트로피 (원본): {entropy_b:.4f}")
    print(f"엔트로피 (정규화): {entropy_b_norm:.4f}")
    print(f"상태: {status_b['level']} - {status_b['description']}")
    print(f"권장: {status_b['recommendation']}\n")

    # ========================================================================
    # Case C: 4가지 구종을 무작위로 섞어 던짐 (높은 Entropy)
    # ========================================================================
    print("=" * 80)
    print("📊 Case C: 4가지 구종 무작위 믹스 (높은 무작위성)")
    print("-" * 80)

    monitor_c = EntropyMonitor(window_size=20)

    # 4가지 구종을 거의 균등하게 섞음
    pitch_types = ['FF', 'SL', 'CH', 'CU']
    sequence_c = []
    for i in range(20):
        pitch_type = pitch_types[i % 4]
        sequence_c.append(pitch_type)
        monitor_c.update(pitch_type)

    print(f"투구 시퀀스: {' -> '.join(sequence_c)}")

    entropy_c = monitor_c.calculate_entropy(normalized=False)
    entropy_c_norm = monitor_c.calculate_entropy(normalized=True)
    status_c = monitor_c.get_predictability_status()
    dist_c = monitor_c.get_pitch_distribution()

    print(f"\n투구 수: {len(monitor_c.pitch_queue)}")
    print(f"구종 분포: {dist_c}")
    print(f"엔트로피 (원본): {entropy_c:.4f} (최대: {math.log2(4):.4f})")
    print(f"엔트로피 (정규화): {entropy_c_norm:.4f}")
    print(f"상태: {status_c['level']} - {status_c['description']}")
    print(f"권장: {status_c['recommendation']}\n")

    # ========================================================================
    # Case D: 실전 시나리오 - 구종 믹스 변화 추적
    # ========================================================================
    print("=" * 80)
    print("📊 Case D: 실전 시나리오 - 투구 패턴 변화 추적")
    print("-" * 80)

    monitor_d = EntropyMonitor(window_size=10)  # 작은 윈도우로 빠른 반응

    # 초반: 직구 위주 (예측 가능)
    print("\n1️⃣ 초반 5투구: 직구 위주")
    for _ in range(4):
        monitor_d.update('FF')
    monitor_d.update('SL')

    entropy_1 = monitor_d.calculate_entropy(normalized=True)
    status_1 = monitor_d.get_predictability_status()
    print(f"   엔트로피: {entropy_1:.4f} - {status_1['description']}")

    # 중반: 구종 다양화
    print("\n2️⃣ 중반 5투구: 구종 다양화")
    for pitch in ['CH', 'CU', 'FF', 'SL', 'CH']:
        monitor_d.update(pitch)

    entropy_2 = monitor_d.calculate_entropy(normalized=True)
    status_2 = monitor_d.get_predictability_status()
    print(f"   엔트로피: {entropy_2:.4f} - {status_2['description']}")
    print(f"   변화: {entropy_2 - entropy_1:+.4f}")

    # 후반: 다시 직구 위주로 회귀 (마무리)
    print("\n3️⃣ 후반 5투구: 직구로 승부")
    for _ in range(5):
        monitor_d.update('FF')

    entropy_3 = monitor_d.calculate_entropy(normalized=True)
    status_3 = monitor_d.get_predictability_status()
    print(f"   엔트로피: {entropy_3:.4f} - {status_3['description']}")
    print(f"   변화: {entropy_3 - entropy_2:+.4f}")

    # ========================================================================
    # 요약
    # ========================================================================
    print("\n" + "=" * 80)
    print("📈 엔트로피 비교 요약")
    print("=" * 80)
    print(f"Case A (직구만):         {entropy_a_norm:.4f} - {status_a['level']}")
    print(f"Case B (2구종 번갈아):   {entropy_b_norm:.4f} - {status_b['level']}")
    print(f"Case C (4구종 균등):     {entropy_c_norm:.4f} - {status_c['level']}")
    print(f"\n이론적 최대값 (4구종): {math.log2(4):.4f} (정규화 시 1.0)")

    print("\n" + "=" * 80)
    print("✅ 완료")
    print("=" * 80)


if __name__ == "__main__":
    main()
