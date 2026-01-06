"""
좌표계 및 항력 방향 검증 스크립트
MLB Statcast 좌표계에서 힘의 방향이 올바른지 확인
"""

import torch
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.physics_engine import SavantPhysicsEngine


def verify_coordinate_system():
    """
    Statcast 좌표계 및 힘의 방향 검증
    """
    print("=" * 80)
    print("🔬 좌표계 및 항력 방향 검증")
    print("=" * 80 + "\n")

    # 물리 엔진 초기화
    engine = SavantPhysicsEngine(
        temperature_f=70.0,
        pressure_hg=29.92,
        humidity_percent=50.0,
        elevation_ft=0.0
    )
    print()

    # ========================================
    # Test 1: 기본 항력 방향 검증
    # ========================================
    print("=" * 80)
    print("Test 1: 항력(Drag Force) 방향 검증")
    print("=" * 80 + "\n")

    print("📌 MLB Statcast 좌표계:")
    print("   - 원점: 홈플레이트 (포수 위치)")
    print("   - +y 방향: 홈플레이트 → 투수판")
    print("   - +z 방향: 수직 상향")
    print("   - +x 방향: 1루 → 3루\n")

    # 투수판에서 홈플레이트 방향으로 던진 공
    # 위치: (0, 18.44m, 1.83m) - 투수판 위치
    # 속도: (0, -42.5m/s, 0) - 홈플레이트 방향 (음수!)
    state = torch.tensor([
        0.0,      # x: 중앙
        18.44,    # y: 투수판 (60.5ft)
        1.83,     # z: 릴리즈 높이 (6ft)
        0.0,      # vx: 옆으로 움직임 없음
        -42.5,    # vy: 홈플레이트 방향 (음수!)
        0.0       # vz: 수직 움직임 없음
    ], dtype=torch.float32)

    # 회전 없음 (순수 항력만 테스트)
    spin_vec = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)

    print("초기 조건:")
    print(f"   위치: ({state[0]:.2f}, {state[1]:.2f}, {state[2]:.2f}) m")
    print(f"   속도: ({state[3]:.2f}, {state[4]:.2f}, {state[5]:.2f}) m/s")
    print(f"   속력: {torch.norm(state[3:6]):.2f} m/s\n")

    # 힘 계산
    forces = engine.compute_forces(state, spin_vec)

    print("계산된 힘:")
    print(f"   총 힘: ({forces[0]:.4f}, {forces[1]:.4f}, {forces[2]:.4f}) N")
    print(f"   Fx (측면): {forces[0]:.4f} N")
    print(f"   Fy (전후): {forces[1]:.4f} N")
    print(f"   Fz (수직): {forces[2]:.4f} N\n")

    # 검증
    print("🔍 검증:")
    print(f"   vy = {state[4]:.2f} m/s (음수 ✓)")
    print(f"   Fy = {forces[1]:.4f} N")

    if state[4] < 0 and forces[1] > 0:
        print("   ✅ PASS: vy < 0 → Fy > 0 (항력이 속도 반대 방향)")
        print("   → 공이 감속됨 (정상 동작)\n")
        test1_pass = True
    else:
        print("   ❌ FAIL: 항력 방향이 잘못되었습니다!")
        print("   → 공이 가속될 수 있음 (로켓 효과)\n")
        test1_pass = False

    # ========================================
    # Test 2: 각 속도 방향에 대한 항력 검증
    # ========================================
    print("=" * 80)
    print("Test 2: 다양한 속도 방향에 대한 항력 검증")
    print("=" * 80 + "\n")

    test_cases = [
        ("홈플레이트 방향 (-y)", torch.tensor([0.0, 18.44, 1.83, 0.0, -40.0, 0.0])),
        ("투수판 방향 (+y)", torch.tensor([0.0, 5.0, 1.83, 0.0, +40.0, 0.0])),
        ("3루 방향 (-x)", torch.tensor([0.0, 18.44, 1.83, -40.0, 0.0, 0.0])),
        ("1루 방향 (+x)", torch.tensor([0.0, 18.44, 1.83, +40.0, 0.0, 0.0])),
        ("상승 (+z)", torch.tensor([0.0, 18.44, 1.83, 0.0, 0.0, +40.0])),
        ("하강 (-z)", torch.tensor([0.0, 18.44, 1.83, 0.0, 0.0, -40.0])),
    ]

    all_tests_pass = True

    for test_name, test_state in test_cases:
        velocity = test_state[3:6]
        forces = engine.compute_forces(test_state, spin_vec)
        drag_force = forces - torch.tensor([0.0, 0.0, -engine.mass * engine.gravity])

        # 항력과 속도의 내적 (음수여야 함)
        dot_product = torch.dot(drag_force, velocity).item()

        print(f"📊 {test_name}:")
        print(f"   속도: ({velocity[0]:+6.1f}, {velocity[1]:+6.1f}, {velocity[2]:+6.1f}) m/s")
        print(f"   항력: ({drag_force[0]:+6.4f}, {drag_force[1]:+6.4f}, {drag_force[2]:+6.4f}) N")
        print(f"   내적: {dot_product:+.6f}")

        if dot_product < 0:
            print(f"   ✅ PASS: 항력이 속도 반대 방향\n")
        else:
            print(f"   ❌ FAIL: 항력이 속도와 같은 방향!\n")
            all_tests_pass = False

    # ========================================
    # Test 3: 실제 투구 시뮬레이션
    # ========================================
    print("=" * 80)
    print("Test 3: 실제 투구 시뮬레이션 (95mph Fastball)")
    print("=" * 80 + "\n")

    # 95mph = 42.5 m/s
    state = torch.tensor([
        0.0,      # x
        18.44,    # y: 투수판
        1.83,     # z: 릴리즈 높이
        0.0,      # vx
        -42.5,    # vy: 홈플레이트 방향
        0.0       # vz
    ], dtype=torch.float32)

    # Backspin (2400 RPM)
    spin_rate_rpm = 2400
    spin_rate_rads = spin_rate_rpm * 2 * 3.14159 / 60
    spin_vec = torch.tensor([spin_rate_rads, 0.0, 0.0], dtype=torch.float32)

    print("초기 조건:")
    print(f"   위치: y={state[1]:.2f}m (투수판)")
    print(f"   속도: vy={state[4]:.2f}m/s (홈플레이트 방향)")
    print(f"   회전: {spin_rate_rpm} RPM (backspin)\n")

    # 0.1초 간격으로 10스텝 시뮬레이션
    dt = 0.1
    current_state = state.clone()

    print("시간     y위치     vy속도    Fy힘     상태")
    print("-" * 60)

    for step in range(11):
        t = step * dt
        forces = engine.compute_forces(current_state, spin_vec)
        accel = forces / engine.mass

        # 속도와 힘의 관계 확인
        vy = current_state[4].item()
        fy = forces[1].item()

        # 상태 체크
        if vy < 0 and fy > 0:
            status = "✓ 감속"
        elif vy < 0 and fy < 0:
            status = "✗ 가속!"
        else:
            status = "?"

        print(f"{t:.2f}s   {current_state[1]:6.2f}m   {vy:+7.2f}m/s   {fy:+7.4f}N   {status}")

        # 다음 스텝 (간단한 Euler 적분)
        current_state[3:6] += accel * dt
        current_state[0:3] += current_state[3:6] * dt

        # 땅에 닿으면 중단
        if current_state[2] < 0:
            break

    print()

    # ========================================
    # 최종 결과
    # ========================================
    print("=" * 80)
    print("📋 검증 결과 요약")
    print("=" * 80 + "\n")

    if test1_pass and all_tests_pass:
        print("✅ 모든 테스트 통과!")
        print("   - 좌표계가 올바르게 구현되었습니다.")
        print("   - 항력이 항상 속도 반대 방향입니다.")
        print("   - 공은 던질수록 감속됩니다. (정상 동작)\n")
    else:
        print("❌ 일부 테스트 실패!")
        print("   - 코드 검토가 필요합니다.\n")

    print("=" * 80)


if __name__ == "__main__":
    verify_coordinate_system()
