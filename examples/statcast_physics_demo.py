"""
통합 예제: Statcast 데이터 + 물리 엔진
실제 MLB 투구 데이터로 물리 시뮬레이션 실행
"""

import torch
import pandas as pd
from pathlib import Path
import sys

# 프로젝트 루트를 path에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data_pipeline.data_loader import AegisDataLoader
from src.physics_engine.savant_physics import SavantPhysicsEngine


def convert_statcast_to_physics(
    pitch_data: pd.Series,
    engine: SavantPhysicsEngine
) -> tuple:
    """
    Statcast 데이터를 물리 엔진 입력 형식으로 변환

    Args:
        pitch_data: Statcast 투구 데이터 (pandas Series)
        engine: 물리 엔진 인스턴스

    Returns:
        tuple: (state_tensor, spin_tensor)
    """
    # Statcast 좌표계: ft 단위
    FT_TO_M = 0.3048

    # 릴리즈 포인트 (ft -> m)
    x = pitch_data['release_pos_x'] * FT_TO_M
    y = pitch_data['release_pos_y'] * FT_TO_M
    z = pitch_data['release_pos_z'] * FT_TO_M

    # 초기 속도 (ft/s -> m/s)
    vx = pitch_data['vx0'] * FT_TO_M
    vy = pitch_data['vy0'] * FT_TO_M
    vz = pitch_data['vz0'] * FT_TO_M

    # 상태 벡터
    state = torch.tensor(
        [x, y, z, vx, vy, vz],
        dtype=torch.float32,
        device=engine.device
    )

    # 회전 속도 (RPM -> rad/s)
    spin_rate = pitch_data['release_spin_rate']
    spin_rads = spin_rate * 2 * 3.14159 / 60

    # 회전 벡터 추정 (간단화: 수직 회전만 고려)
    # 실제로는 더 복잡한 계산 필요
    spin_vec = torch.tensor(
        [spin_rads, 0.0, 0.0],  # 주로 backspin 가정
        dtype=torch.float32,
        device=engine.device
    )

    return state, spin_vec


def analyze_pitch(
    pitch_data: pd.Series,
    engine: SavantPhysicsEngine
) -> dict:
    """
    단일 투구 분석

    Args:
        pitch_data: Statcast 투구 데이터
        engine: 물리 엔진

    Returns:
        dict: 분석 결과
    """
    # 데이터 변환
    state, spin_vec = convert_statcast_to_physics(pitch_data, engine)

    # 힘 계산
    forces = engine.compute_forces(state, spin_vec)
    accel = engine.get_acceleration(state, spin_vec)

    # 속도 계산
    v_mag = torch.norm(state[3:6]).item()
    v_mph = v_mag * 2.237  # m/s -> mph

    # Spin Factor 계산
    omega_mag = torch.norm(spin_vec).item()
    spin_factor = engine.compute_spin_factor(
        torch.tensor(v_mag), torch.tensor(omega_mag)
    ).item()

    # Coefficients
    c_l = engine.compute_lift_coefficient(torch.tensor(spin_factor)).item()
    c_d = engine.compute_drag_coefficient(torch.tensor(spin_factor)).item()

    return {
        'pitch_type': pitch_data['pitch_type'],
        'velocity_mph': v_mph,
        'spin_rpm': pitch_data['release_spin_rate'],
        'spin_factor': spin_factor,
        'lift_coef': c_l,
        'drag_coef': c_d,
        'total_force': torch.norm(forces).item(),
        'vertical_force': forces[2].item(),
        'vertical_accel': accel[2].item(),
        'pitcher': pitch_data['pitcher'],
    }


def main():
    print("=" * 80)
    print("🎯 Statcast 데이터 + 물리 엔진 통합 분석")
    print("=" * 80 + "\n")

    # 1. 데이터 로더 초기화
    print("📊 데이터 로드...")
    with AegisDataLoader() as loader:
        # 2024년 데이터 100개 샘플
        df = loader.load_data_by_year(year=2024, limit=100)
    print()

    if df.empty:
        print("❌ 데이터를 찾을 수 없습니다.")
        return

    # 2. 물리 엔진 초기화 (표준 조건)
    print("⚙️  물리 엔진 초기화...")
    engine = SavantPhysicsEngine(
        temperature_f=70.0,
        pressure_hg=29.92,
        humidity_percent=50.0,
        elevation_ft=0.0
    )
    print()

    # 3. 투구 타입별 분석
    print("=" * 80)
    print("📈 투구 타입별 물리 분석")
    print("=" * 80 + "\n")

    # 투구 타입별 그룹화
    pitch_types = df['pitch_type'].value_counts().head(5).index.tolist()

    results = []

    for pitch_type in pitch_types:
        # 해당 타입의 투구 선택 (첫 번째만)
        pitch_samples = df[df['pitch_type'] == pitch_type].head(1)

        for _, pitch in pitch_samples.iterrows():
            result = analyze_pitch(pitch, engine)
            results.append(result)

            print(f"🎾 {result['pitch_type']} (투수 {result['pitcher']})")
            print(f"   속도: {result['velocity_mph']:.1f} mph")
            print(f"   회전: {result['spin_rpm']:.0f} RPM")
            print(f"   Spin Factor: {result['spin_factor']:.4f}")
            print(f"   C_L: {result['lift_coef']:.4f}, C_D: {result['drag_coef']:.4f}")
            print(f"   수직력: {result['vertical_force']:.3f} N")
            print(f"   수직 가속도: {result['vertical_accel']:.2f} m/s²")
            print()

    # 4. 통계 요약
    print("=" * 80)
    print("📊 통계 요약")
    print("=" * 80 + "\n")

    results_df = pd.DataFrame(results)

    print("투구 타입별 평균:")
    summary = results_df.groupby('pitch_type').agg({
        'velocity_mph': 'mean',
        'spin_rpm': 'mean',
        'spin_factor': 'mean',
        'lift_coef': 'mean',
        'drag_coef': 'mean',
        'vertical_force': 'mean'
    }).round(3)

    print(summary.to_string())
    print()

    # 5. 환경별 비교 (같은 투구, 다른 환경)
    print("=" * 80)
    print("🌡️  환경 조건별 비교 (동일 투구)")
    print("=" * 80 + "\n")

    # 첫 번째 투구 선택
    sample_pitch = df.iloc[0]

    environments = [
        ("해수면 표준", 70.0, 29.92, 50.0, 0.0),
        ("더운 날씨", 95.0, 29.80, 80.0, 0.0),
        ("Coors Field", 75.0, 24.60, 30.0, 5280.0),
    ]

    for env_name, temp, pressure, humidity, elevation in environments:
        engine_env = SavantPhysicsEngine(
            temperature_f=temp,
            pressure_hg=pressure,
            humidity_percent=humidity,
            elevation_ft=elevation
        )

        result = analyze_pitch(sample_pitch, engine_env)

        print(f"📍 {env_name}:")
        print(f"   공기 밀도: {engine_env.air_density:.4f} kg/m³")
        print(f"   수직력: {result['vertical_force']:.3f} N")
        print(f"   총 힘: {result['total_force']:.3f} N")
        print()

    print("=" * 80)
    print("✅ 분석 완료")
    print("=" * 80)


if __name__ == "__main__":
    main()
