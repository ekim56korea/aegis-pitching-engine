"""
SavantPhysicsEngine: MLB Statcast 호환 고정밀 야구 물리 엔진
Alan Nathan Model 기반 고급 공기역학 구현
"""

import torch
from typing import Tuple, Optional
import math


class SavantPhysicsEngine:
    """
    MLB Statcast 데이터와 호환되는 고정밀 물리 엔진

    Features:
        - Alan Nathan Model 기반 공기역학
        - 환경 변수에 따른 동적 공기 밀도 계산
        - Spin saturation을 고려한 Lift/Drag Coefficient
        - PyTorch 기반 배치 처리 지원

    Coordinate System (Statcast):
        - x: 홈플레이트에서 3루 방향 (ft)
        - y: 홈플레이트에서 투수판 방향 (ft)
        - z: 수직 상향 (ft)
    """

    # Physical Constants
    CONSTANTS = {
        'mass': 0.145,              # kg (야구공 질량)
        'diameter': 0.074,          # m (야구공 지름)
        'radius': 0.037,            # m (야구공 반지름)
        'circumference': 0.232,     # m (야구공 둘레, ~9.125 inches)
        'gravity': 9.80665,         # m/s^2 (중력 가속도)
        'R_air': 287.05,           # J/(kg·K) (공기의 기체 상수)
    }

    # 단면적 계산 (πr²)
    CONSTANTS['area'] = math.pi * (CONSTANTS['radius'] ** 2)  # m^2

    # Drag coefficient 기본값
    CD0 = 0.40  # 무회전 시 항력 계수
    CD_SPIN = 0.05  # 스핀에 의한 추가 항력 계수

    def __init__(
        self,
        temperature_f: float = 70.0,
        pressure_hg: float = 29.92,
        humidity_percent: float = 50.0,
        elevation_ft: float = 0.0,
        device: str = 'cpu'
    ):
        """
        물리 엔진 초기화

        Args:
            temperature_f: 온도 (화씨, °F)
            pressure_hg: 기압 (수은주 인치, inHg)
            humidity_percent: 상대 습도 (%)
            elevation_ft: 고도 (피트, ft)
            device: PyTorch 디바이스 ('cpu' 또는 'cuda')
        """
        self.device = torch.device(device)

        # 환경 변수 저장
        self.temperature_f = temperature_f
        self.pressure_hg = pressure_hg
        self.humidity_percent = humidity_percent
        self.elevation_ft = elevation_ft

        # 공기 밀도 계산
        self.air_density = self._calculate_air_density(
            temperature_f, pressure_hg, humidity_percent, elevation_ft
        )

        # 상수를 Tensor로 변환
        self.mass = torch.tensor(
            self.CONSTANTS['mass'], dtype=torch.float32, device=self.device
        )
        self.area = torch.tensor(
            self.CONSTANTS['area'], dtype=torch.float32, device=self.device
        )
        self.radius = torch.tensor(
            self.CONSTANTS['radius'], dtype=torch.float32, device=self.device
        )
        self.gravity = torch.tensor(
            self.CONSTANTS['gravity'], dtype=torch.float32, device=self.device
        )
        self.rho = torch.tensor(
            self.air_density, dtype=torch.float32, device=self.device
        )

        print(f"✅ SavantPhysicsEngine 초기화 완료")
        print(f"   온도: {temperature_f:.1f}°F, 기압: {pressure_hg:.2f}inHg")
        print(f"   습도: {humidity_percent:.1f}%, 고도: {elevation_ft:.0f}ft")
        print(f"   공기 밀도: {self.air_density:.4f} kg/m³")

    def _calculate_air_density(
        self,
        temp_f: float,
        pressure_hg: float,
        humidity: float,
        elevation: float
    ) -> float:
        """
        환경 조건에 따른 공기 밀도 계산

        Args:
            temp_f: 온도 (°F)
            pressure_hg: 기압 (inHg)
            humidity: 상대 습도 (%)
            elevation: 고도 (ft)

        Returns:
            float: 공기 밀도 (kg/m³)

        Notes:
            - 이상 기체 법칙 사용: ρ = P/(R·T)
            - 습도와 고도 효과 보정 포함
        """
        # 단위 변환
        temp_k = (temp_f - 32) * 5/9 + 273.15  # °F -> K
        pressure_pa = pressure_hg * 3386.39  # inHg -> Pa

        # 고도에 따른 기압 보정 (해발 1000ft당 약 3.5% 감소)
        pressure_pa *= (1 - 0.0000225577 * elevation * 0.3048) ** 5.25588

        # 포화 수증기압 계산 (Magnus formula)
        temp_c = temp_k - 273.15
        e_sat = 611.2 * math.exp(17.67 * temp_c / (temp_c + 243.5))  # Pa

        # 실제 수증기압
        e_actual = e_sat * (humidity / 100.0)

        # 건조 공기 압력
        p_dry = pressure_pa - e_actual

        # 공기 밀도 계산 (습한 공기)
        # ρ = (p_dry/(R_dry·T)) + (e/(R_vapor·T))
        R_dry = 287.05  # J/(kg·K)
        R_vapor = 461.5  # J/(kg·K)

        rho = (p_dry / (R_dry * temp_k)) + (e_actual / (R_vapor * temp_k))

        return rho

    def compute_spin_factor(
        self,
        velocity: torch.Tensor,
        spin_rate: torch.Tensor
    ) -> torch.Tensor:
        """
        Spin Factor 계산: S = (r·ω)/v

        Args:
            velocity: 속도 벡터 크기 (m/s) - shape: (batch_size,) or scalar
            spin_rate: 회전 속도 크기 (rad/s) - shape: (batch_size,) or scalar

        Returns:
            torch.Tensor: Spin Factor (무차원)
        """
        # v_tangential = r * ω
        v_tangential = self.radius * spin_rate

        # S = v_tangential / v
        # 0으로 나누기 방지
        spin_factor = v_tangential / (velocity + 1e-6)

        return spin_factor

    def compute_lift_coefficient(
        self,
        spin_factor: torch.Tensor
    ) -> torch.Tensor:
        """
        Lift Coefficient 계산 (Alan Nathan Model with Spin Saturation)

        Args:
            spin_factor: Spin Factor S = (r·ω)/v

        Returns:
            torch.Tensor: Lift Coefficient C_L

        Notes:
            C_L = 1 / (2.32 + 0.4/S)
            - 낮은 S: Lift가 작음 (스핀이 약함)
            - 높은 S: Lift가 포화됨 (스핀 효과 한계)
        """
        # Spin saturation을 고려한 비선형 모델
        # C_L = 1 / (2.32 + 0.4/S)
        c_l = 1.0 / (2.32 + 0.4 / (spin_factor + 1e-6))

        return c_l

    def compute_drag_coefficient(
        self,
        spin_factor: torch.Tensor
    ) -> torch.Tensor:
        """
        Drag Coefficient 계산 (스핀 의존성 포함)

        Args:
            spin_factor: Spin Factor S = (r·ω)/v

        Returns:
            torch.Tensor: Drag Coefficient C_D

        Notes:
            C_D = C_D0 + C_D_spin * S
            - 스핀이 증가하면 항력도 증가
        """
        c_d = self.CD0 + self.CD_SPIN * spin_factor

        return c_d

    def compute_forces(
        self,
        state: torch.Tensor,
        spin_vec: torch.Tensor
    ) -> torch.Tensor:
        """
        야구공에 작용하는 모든 힘을 계산

        Args:
            state: 상태 벡터 (x, y, z, vx, vy, vz) - shape: (batch_size, 6) or (6,)
                   Statcast 좌표계 (ft 단위를 m로 변환해서 입력해야 함)
            spin_vec: 회전 벡터 (ωx, ωy, ωz) in rad/s - shape: (batch_size, 3) or (3,)

        Returns:
            torch.Tensor: 알짜 힘 벡터 (Fx, Fy, Fz) in N - shape: (batch_size, 3) or (3,)

        Notes:
            Total Force = Gravity + Drag + Magnus
            - Gravity: F_g = (0, 0, -mg)
            - Drag: F_d = -0.5 * ρ * A * C_D * |v| * v
            - Magnus: F_m = 0.5 * ρ * A * C_L * (ω × v) / |ω|
        """
        # 상태 분리
        if state.dim() == 1:
            # Single sample
            velocity = state[3:6]  # (vx, vy, vz)
        else:
            # Batch
            velocity = state[:, 3:6]  # (batch_size, 3)

        # 속도 및 회전 크기
        v_mag = torch.norm(velocity, dim=-1, keepdim=True)  # (batch_size, 1) or (1,)
        omega_mag = torch.norm(spin_vec, dim=-1, keepdim=True)  # (batch_size, 1) or (1,)

        # Spin Factor
        spin_factor = self.compute_spin_factor(
            v_mag.squeeze(-1), omega_mag.squeeze(-1)
        )

        # Coefficients
        c_l = self.compute_lift_coefficient(spin_factor)
        c_d = self.compute_drag_coefficient(spin_factor)

        # 1. Gravity Force: F_g = (0, 0, -mg)
        if state.dim() == 1:
            f_gravity = torch.tensor(
                [0.0, 0.0, -self.mass * self.gravity],
                dtype=torch.float32,
                device=self.device
            )
        else:
            batch_size = state.shape[0]
            f_gravity = torch.zeros(batch_size, 3, device=self.device)
            f_gravity[:, 2] = -self.mass * self.gravity

        # 2. Drag Force: F_d = -0.5 * ρ * A * C_D * |v| * v
        drag_magnitude = 0.5 * self.rho * self.area * c_d.unsqueeze(-1) * v_mag
        f_drag = -drag_magnitude * velocity / (v_mag + 1e-6)

        # 3. Magnus Force: F_m = 0.5 * ρ * A * C_L * (ω × v) / |ω|
        # 외적: ω × v
        omega_cross_v = torch.cross(spin_vec, velocity, dim=-1)

        # Magnus force magnitude
        magnus_magnitude = 0.5 * self.rho * self.area * c_l.unsqueeze(-1)

        # Magnus force 방향 (normalized ω × v)
        f_magnus = magnus_magnitude * omega_cross_v / (omega_mag + 1e-6)

        # Total Force
        f_total = f_gravity + f_drag + f_magnus

        return f_total

    def get_acceleration(
        self,
        state: torch.Tensor,
        spin_vec: torch.Tensor
    ) -> torch.Tensor:
        """
        가속도 계산 (a = F/m)

        Args:
            state: 상태 벡터 (x, y, z, vx, vy, vz)
            spin_vec: 회전 벡터 (ωx, ωy, ωz)

        Returns:
            torch.Tensor: 가속도 벡터 (ax, ay, az) in m/s²
        """
        forces = self.compute_forces(state, spin_vec)
        acceleration = forces / self.mass

        return acceleration

    def __repr__(self) -> str:
        return (
            f"SavantPhysicsEngine(\n"
            f"  temperature={self.temperature_f:.1f}°F,\n"
            f"  pressure={self.pressure_hg:.2f}inHg,\n"
            f"  humidity={self.humidity_percent:.1f}%,\n"
            f"  elevation={self.elevation_ft:.0f}ft,\n"
            f"  air_density={self.air_density:.4f}kg/m³,\n"
            f"  device={self.device}\n"
            f")"
        )


def main():
    """사용 예시 및 테스트"""
    print("=" * 80)
    print("🚀 SavantPhysicsEngine 테스트")
    print("=" * 80 + "\n")

    # 1. 다양한 환경 조건에서 엔진 생성
    print("📊 환경별 공기 밀도 비교:\n")

    # 해수면 표준 조건
    engine_standard = SavantPhysicsEngine(
        temperature_f=70.0,
        pressure_hg=29.92,
        humidity_percent=50.0,
        elevation_ft=0.0
    )
    print()

    # 더운 날씨 (습함)
    engine_hot = SavantPhysicsEngine(
        temperature_f=95.0,
        pressure_hg=29.80,
        humidity_percent=80.0,
        elevation_ft=0.0
    )
    print()

    # 고지대 (Coors Field, Denver - 5,280ft)
    engine_coors = SavantPhysicsEngine(
        temperature_f=75.0,
        pressure_hg=24.60,
        humidity_percent=30.0,
        elevation_ft=5280.0
    )
    print()

    # 2. 힘 계산 테스트 (단일 샘플)
    print("=" * 80)
    print("🔬 힘 계산 테스트 (4-Seam Fastball)")
    print("=" * 80 + "\n")

    # 초기 상태 (Statcast 단위: ft -> m 변환)
    # 투구 시점: 릴리즈 포인트
    # 위치: (0, 60.5ft, 6ft) -> (0, 18.44m, 1.83m)
    # 속도: 95mph -> 42.5m/s (y 방향)
    state = torch.tensor([
        0.0,      # x position (m)
        18.44,    # y position (m) - 투수판에서
        1.83,     # z position (m) - 릴리즈 높이
        0.0,      # vx (m/s)
        -42.5,    # vy (m/s) - 홈플레이트 방향 (음수)
        0.0       # vz (m/s)
    ], dtype=torch.float32)

    # 회전 벡터 (2400 RPM backspin)
    # 2400 RPM = 2400 * 2π / 60 = 251.3 rad/s
    spin_rate_rpm = 2400
    spin_rate_rads = spin_rate_rpm * 2 * math.pi / 60

    # Backspin (x 축 회전)
    spin_vec = torch.tensor([
        spin_rate_rads,  # ωx (backspin)
        0.0,             # ωy
        0.0              # ωz
    ], dtype=torch.float32)

    # 힘 계산
    forces = engine_standard.compute_forces(state, spin_vec)

    print(f"초기 조건:")
    print(f"  위치: ({state[0]:.2f}, {state[1]:.2f}, {state[2]:.2f}) m")
    print(f"  속도: ({state[3]:.2f}, {state[4]:.2f}, {state[5]:.2f}) m/s")
    print(f"  속력: {torch.norm(state[3:6]):.2f} m/s ({torch.norm(state[3:6]) * 2.237:.1f} mph)")
    print(f"  회전: {spin_rate_rpm} RPM (backspin)\n")

    print(f"작용하는 힘:")
    print(f"  총 힘: ({forces[0]:.4f}, {forces[1]:.4f}, {forces[2]:.4f}) N")
    print(f"  수직 성분: {forces[2]:.4f} N (양수 = 상승력)\n")

    # 가속도 계산
    accel = engine_standard.get_acceleration(state, spin_vec)
    print(f"가속도:")
    print(f"  ({accel[0]:.2f}, {accel[1]:.2f}, {accel[2]:.2f}) m/s²")
    print(f"  수직 가속도: {accel[2]:.2f} m/s² (중력: -9.81 m/s²)\n")

    # 3. 배치 처리 테스트
    print("=" * 80)
    print("📦 배치 처리 테스트 (3개 투구)")
    print("=" * 80 + "\n")

    # 여러 투구 (Fastball, Slider, Curveball)
    batch_states = torch.tensor([
        [0.0, 18.44, 1.83, 0.0, -42.5, 0.0],   # Fastball
        [0.0, 18.44, 1.83, 0.0, -38.0, 0.0],   # Slider
        [0.0, 18.44, 1.83, 0.0, -35.0, 0.0],   # Curveball
    ], dtype=torch.float32)

    batch_spins = torch.tensor([
        [251.3, 0.0, 0.0],      # Fastball: 2400 RPM backspin
        [150.0, 0.0, 220.0],    # Slider: topspin + sidespin
        [0.0, 0.0, -280.0],     # Curveball: 2700 RPM topspin
    ], dtype=torch.float32)

    batch_forces = engine_standard.compute_forces(batch_states, batch_spins)

    pitch_types = ['Fastball', 'Slider', 'Curveball']
    for i, pitch_type in enumerate(pitch_types):
        print(f"{pitch_type}:")
        print(f"  힘: ({batch_forces[i, 0]:.3f}, {batch_forces[i, 1]:.3f}, {batch_forces[i, 2]:.3f}) N")
        print(f"  수직 힘: {batch_forces[i, 2]:.3f} N\n")

    print("=" * 80)
    print("✅ 테스트 완료")
    print("=" * 80)


if __name__ == "__main__":
    main()
