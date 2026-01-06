"""
TunnelingAnalyzer: 투구 터널링 효과 분석 도구
실제 투구와 반사실적(Counterfactual) 궤적을 비교하여 터널링 점수 계산
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Optional
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.physics_engine import SavantPhysicsEngine
from src.data_pipeline import AegisDataLoader


class TunnelingAnalyzer:
    """
    투구 터널링(Tunneling) 분석 도구 - Production Version

    Features:
        - 투수별 구종별 평균 DNA 추출 (get_pitch_profile)
        - Delta 주입 방식의 반사실적 투구 시뮬레이션
        - Decision Point에서의 궤적 차이 계산
        - VAA/HAA 계산 (Approach Angles)
        - Tunnel Score 계산
        - 타자 시점 시각화
    """

    # 투구 타입별 기본 특성 (Fallback용)
    PITCH_TYPE_PROFILES = {
        'FF': {  # 4-Seam Fastball
            'spin_rate': 2300,
            'spin_axis_x': 1.0,  # Backspin
            'spin_axis_y': 0.0,
            'spin_axis_z': 0.0,
            'velocity_modifier': 1.0,
        },
        'SI': {  # Sinker
            'spin_rate': 2150,
            'spin_axis_x': 0.8,
            'spin_axis_y': 0.0,
            'spin_axis_z': -0.6,  # Sidespin
            'velocity_modifier': 0.98,
        },
        'FC': {  # Cutter
            'spin_rate': 2400,
            'spin_axis_x': 0.8,
            'spin_axis_y': 0.0,
            'spin_axis_z': 0.6,
            'velocity_modifier': 0.96,
        },
        'SL': {  # Slider
            'spin_rate': 2500,
            'spin_axis_x': 0.5,
            'spin_axis_y': 0.0,
            'spin_axis_z': 0.866,  # 주로 sidespin
            'velocity_modifier': 0.90,
        },
        'CU': {  # Curveball
            'spin_rate': 2650,
            'spin_axis_x': 0.0,
            'spin_axis_y': 0.0,
            'spin_axis_z': -1.0,  # Topspin
            'velocity_modifier': 0.83,
        },
        'CH': {  # Changeup
            'spin_rate': 1800,
            'spin_axis_x': 0.7,
            'spin_axis_y': 0.0,
            'spin_axis_z': -0.7,
            'velocity_modifier': 0.88,
        },
    }

    # Decision Point: 투구 후 0.167초 (약 23.8ft)
    DECISION_TIME = 0.167  # seconds

    def __init__(
        self,
        data_loader: Optional[AegisDataLoader] = None,
        physics_engine: Optional[SavantPhysicsEngine] = None,
        dt: float = 0.001  # 시뮬레이션 시간 간격
    ):
        """
        TunnelingAnalyzer 초기화

        Args:
            data_loader: 데이터 로더 (None이면 새로 생성)
            physics_engine: 물리 엔진 (None이면 표준 조건으로 생성)
            dt: 시뮬레이션 시간 간격 (초)
        """
        self.data_loader = data_loader

        if physics_engine is None:
            self.engine = SavantPhysicsEngine(
                temperature_f=70.0,
                pressure_hg=29.92,
                humidity_percent=50.0,
                elevation_ft=0.0
            )
        else:
            self.engine = physics_engine

        self.dt = dt

        print(f"✅ TunnelingAnalyzer 초기화 (Production Version)")
        print(f"   시뮬레이션 간격: {dt*1000:.1f}ms")
        print(f"   Decision Point: {self.DECISION_TIME*1000:.1f}ms")

    def get_pitch_profile(
        self,
        pitcher_id: int,
        pitch_type: str
    ) -> Dict[str, np.ndarray]:
        """
        투수별 구종별 평균 DNA 추출

        Args:
            pitcher_id: 투수 ID
            pitch_type: 구종 ('FF', 'SI', 'FC', 'SL', 'CU', 'CH', etc.)

        Returns:
            dict: {
                'release_pos': [3] - (x, y, z) Extension 포함
                'release_vel': [3] - (vx, vy, vz) Launch Angle 내포
                'spin_rate': float - RPM
                'spin_axis': float - Degree (0-360)
                'avg_plate_speed': float - mph (검증용)
            }
        """
        # 새로운 data loader 인스턴스 생성 (connection 재사용 문제 방지)
        with AegisDataLoader() as loader:
            df = loader.load_pitcher_data(pitcher_id)

        if df.empty:
            raise ValueError(f"No data found for pitcher_id={pitcher_id}")

        # 구종 필터링
        pitch_df = df[df['pitch_type'] == pitch_type]

        if pitch_df.empty:
            # Fallback: 기본 프로파일 사용
            print(f"⚠️  투수 {pitcher_id}의 {pitch_type} 데이터 없음. 기본 프로파일 사용.")
            if pitch_type not in self.PITCH_TYPE_PROFILES:
                raise ValueError(f"Unknown pitch type: {pitch_type}")

            # 기본값 반환 (임의로 설정)
            return {
                'release_pos': np.array([0.0, 18.44, 1.8]),  # 60.5ft = 18.44m
                'release_vel': np.array([0.0, -40.0, 0.0]),  # 대략 90mph
                'spin_rate': self.PITCH_TYPE_PROFILES[pitch_type]['spin_rate'],
                'spin_axis': 180.0,  # backspin
                'avg_plate_speed': 90.0
            }

        # 평균 계산
        FT_TO_M = 0.3048

        release_pos = np.array([
            pitch_df['release_pos_x'].mean() * FT_TO_M,
            pitch_df['release_pos_y'].mean() * FT_TO_M,
            pitch_df['release_pos_z'].mean() * FT_TO_M
        ])

        release_vel = np.array([
            pitch_df['vx0'].mean() * FT_TO_M,
            pitch_df['vy0'].mean() * FT_TO_M,
            pitch_df['vz0'].mean() * FT_TO_M
        ])

        spin_rate = pitch_df['release_spin_rate'].mean()

        # spin_axis 계산 (ax, ay, az -> degree)
        # Statcast에서는 ax, ay 값으로 spin axis 추정 가능
        # 간단화: spin_axis = arctan2(ax, ay) 형태로 계산
        # 여기서는 평균 ax, ay 사용
        if 'ax' in pitch_df.columns and 'ay' in pitch_df.columns:
            ax_mean = pitch_df['ax'].mean()
            ay_mean = pitch_df['ay'].mean()
            spin_axis = np.degrees(np.arctan2(ax_mean, ay_mean)) % 360
        else:
            # Fallback
            spin_axis = 180.0  # backspin

        avg_plate_speed = pitch_df['release_speed'].mean()

        profile = {
            'release_pos': release_pos,
            'release_vel': release_vel,
            'spin_rate': spin_rate,
            'spin_axis': spin_axis,
            'avg_plate_speed': avg_plate_speed
        }

        print(f"✅ Pitch Profile 추출: Pitcher {pitcher_id}, Type {pitch_type}")
        print(f"   Position: {release_pos}")
        print(f"   Velocity: {release_vel}")
        print(f"   Spin Rate: {spin_rate:.0f} RPM")
        print(f"   Spin Axis: {spin_axis:.1f}°")

        return profile


    def _convert_statcast_to_state(
        self,
        pitch_data: pd.Series
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Statcast 데이터를 물리 엔진 입력으로 변환

        Args:
            pitch_data: Statcast 투구 데이터 (pandas Series)

        Returns:
            tuple: (initial_state, spin_vec)
        """
        FT_TO_M = 0.3048

        # 초기 위치 (ft -> m)
        x0 = pitch_data['release_pos_x'] * FT_TO_M
        y0 = pitch_data['release_pos_y'] * FT_TO_M
        z0 = pitch_data['release_pos_z'] * FT_TO_M

        # 초기 속도 (ft/s -> m/s)
        vx0 = pitch_data['vx0'] * FT_TO_M
        vy0 = pitch_data['vy0'] * FT_TO_M
        vz0 = pitch_data['vz0'] * FT_TO_M

        initial_state = torch.tensor(
            [x0, y0, z0, vx0, vy0, vz0],
            dtype=torch.float32,
            device=self.engine.device
        )

        # 회전 (RPM -> rad/s)
        spin_rate = pitch_data['release_spin_rate']
        spin_rads = spin_rate * 2 * np.pi / 60

        # 간단화: backspin 가정 (실제로는 spin axis 필요)
        spin_vec = torch.tensor(
            [spin_rads, 0.0, 0.0],
            dtype=torch.float32,
            device=self.engine.device
        )

        return initial_state, spin_vec

    def _simulate_trajectory(
        self,
        initial_state: torch.Tensor,
        spin_vec: torch.Tensor,
        max_time: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        궤적 시뮬레이션 (Euler 적분)

        Args:
            initial_state: 초기 상태 [x, y, z, vx, vy, vz]
            spin_vec: 회전 벡터 [ωx, ωy, ωz]
            max_time: 최대 시뮬레이션 시간

        Returns:
            tuple: (time_array, trajectory_array)
                - time_array: [N] 시간 배열
                - trajectory_array: [N, 6] 상태 배열 (x, y, z, vx, vy, vz)
        """
        current_state = initial_state.clone()
        trajectory = [current_state.cpu().numpy()]
        time_points = [0.0]

        t = 0.0
        while t < max_time:
            # 현재 상태로 힘 계산
            forces = self.engine.compute_forces(
                current_state.unsqueeze(0),
                spin_vec.unsqueeze(0)
            ).squeeze(0)

            # 가속도
            accel = forces / self.engine.mass

            # 상태 업데이트 (Euler method)
            current_state[3:6] += accel * self.dt  # 속도 업데이트
            current_state[0:3] += current_state[3:6] * self.dt  # 위치 업데이트

            t += self.dt

            # 저장
            trajectory.append(current_state.cpu().numpy())
            time_points.append(t)

            # 땅에 닿으면 중단 (z < 0)
            if current_state[2] < 0:
                break

            # 홈플레이트를 지나면 중단 (y < 0)
            if current_state[1] < 0:
                break

        trajectory_array = np.array(trajectory)
        time_array = np.array(time_points)

        return time_array, trajectory_array

    def calculate_approach_angles(
        self,
        trajectory: np.ndarray
    ) -> Dict[str, float]:
        """
        궤적 마지막 지점(홈플레이트)에서의 접근 각도 계산

        Args:
            trajectory: [N, 6] 궤적 배열 (x, y, z, vx, vy, vz)

        Returns:
            dict: {
                'vaa': Vertical Approach Angle (도),
                'haa': Horizontal Approach Angle (도)
            }
        """
        # 마지막 지점의 속도 벡터
        final_velocity = trajectory[-1, 3:6]  # [vx, vy, vz]

        vx_f, vy_f, vz_f = final_velocity

        # VAA = arctan(vz / vy)
        # 음수: 하강, 양수: 상승
        vaa_rad = np.arctan2(vz_f, -vy_f)  # -vy because vy is negative (toward home)
        vaa_deg = np.degrees(vaa_rad)

        # HAA = arctan(vx / vy)
        # 음수: 좌측, 양수: 우측 (투수 시점)
        haa_rad = np.arctan2(vx_f, -vy_f)
        haa_deg = np.degrees(haa_rad)

        return {
            'vaa': vaa_deg,
            'haa': haa_deg
        }

    def simulate_counterfactual(
        self,
        actual_pitch_data: pd.Series,
        target_pitch_type: str,
        pitcher_id: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        반사실적(Counterfactual) 투구 시뮬레이션 - Delta Injection Method

        실제 투구와 동일한 타이밍/컨디션에서, 구종만 target_pitch_type으로 변경.
        Delta_Pos, Delta_Vel, Delta_Spin을 계산하여 실제 투구에 주입.

        Args:
            actual_pitch_data: 실제 투구 데이터 (Statcast)
            target_pitch_type: 목표 투구 타입 ('FF', 'SL', 'CU', etc.)
            pitcher_id: 투수 ID (Profile 추출용, None이면 기본 프로파일 사용)

        Returns:
            dict: {
                'actual_time': [N] 실제 투구 시간,
                'actual_traj': [N, 6] 실제 투구 궤적,
                'cf_time': [M] 반사실적 시간,
                'cf_traj': [M, 6] 반사실적 궤적,
                'initial_state': [6] 초기 상태,
                'actual_spin': [3] 실제 회전,
                'cf_spin': [3] 반사실적 회전,
                'actual_vaa': float - 실제 VAA (도),
                'cf_vaa': float - 반사실적 VAA (도),
                'actual_haa': float - 실제 HAA (도),
                'cf_haa': float - 반사실적 HAA (도)
            }
        """
        # 1. 실제 투구 시뮬레이션
        initial_state, actual_spin = self._convert_statcast_to_state(actual_pitch_data)
        actual_time, actual_traj = self._simulate_trajectory(initial_state, actual_spin)

        # 2. Profile 추출 (투수별 구종별 평균 DNA)
        if pitcher_id is not None:
            try:
                # 실제 투구 프로파일
                actual_pitch_type = actual_pitch_data.get('pitch_type', 'FF')
                actual_profile = self.get_pitch_profile(pitcher_id, actual_pitch_type)

                # 목표 투구 프로파일
                target_profile = self.get_pitch_profile(pitcher_id, target_pitch_type)

            except Exception as e:
                print(f"⚠️  Profile 추출 실패: {e}. Fallback 사용.")
                actual_profile = None
                target_profile = None
        else:
            actual_profile = None
            target_profile = None

        # 3. Delta 계산
        FT_TO_M = 0.3048

        if actual_profile is not None and target_profile is not None:
            # Profile 기반 Delta 계산
            delta_pos = target_profile['release_pos'] - actual_profile['release_pos']
            delta_vel = target_profile['release_vel'] - actual_profile['release_vel']

            # Spin Delta
            # Spin axis (degree -> radian, then to vector)
            actual_spin_axis_rad = np.radians(actual_profile['spin_axis'])
            target_spin_axis_rad = np.radians(target_profile['spin_axis'])

            actual_spin_rate_rads = actual_profile['spin_rate'] * 2 * np.pi / 60
            target_spin_rate_rads = target_profile['spin_rate'] * 2 * np.pi / 60

            # Spin vector (simplified: assume spin axis in x-z plane with tilt angle)
            # spin_axis: 0° = pure backspin (+x), 90° = sidespin (+z), 180° = topspin (-x)
            actual_spin_vec = np.array([
                actual_spin_rate_rads * np.cos(actual_spin_axis_rad),
                0.0,
                actual_spin_rate_rads * np.sin(actual_spin_axis_rad)
            ])

            target_spin_vec = np.array([
                target_spin_rate_rads * np.cos(target_spin_axis_rad),
                0.0,
                target_spin_rate_rads * np.sin(target_spin_axis_rad)
            ])

            delta_spin = target_spin_vec - actual_spin_vec

            print(f"📊 Delta Injection:")
            print(f"   ΔPos: {delta_pos}")
            print(f"   ΔVel: {delta_vel}")
            print(f"   ΔSpin: {delta_spin}")

        else:
            # Fallback: 기본 PITCH_TYPE_PROFILES 사용
            if target_pitch_type not in self.PITCH_TYPE_PROFILES:
                raise ValueError(f"Unknown pitch type: {target_pitch_type}")

            profile = self.PITCH_TYPE_PROFILES[target_pitch_type]

            # 속도 조정
            velocity_vector = initial_state[3:6].cpu().numpy()
            velocity_mag = np.linalg.norm(velocity_vector)
            new_velocity_mag = velocity_mag * profile['velocity_modifier']
            new_velocity_vector = velocity_vector / velocity_mag * new_velocity_mag

            delta_pos = np.zeros(3)
            delta_vel = new_velocity_vector - velocity_vector

            # Spin Delta
            cf_spin_rate = profile['spin_rate'] * 2 * np.pi / 60  # RPM -> rad/s
            cf_spin_axis = np.array([
                profile['spin_axis_x'],
                profile['spin_axis_y'],
                profile['spin_axis_z']
            ])
            cf_spin_axis = cf_spin_axis / (np.linalg.norm(cf_spin_axis) + 1e-6)
            cf_spin_vec = cf_spin_axis * cf_spin_rate

            actual_spin_np = actual_spin.cpu().numpy()
            delta_spin = cf_spin_vec - actual_spin_np

        # 4. Delta 주입으로 Counterfactual 생성
        cf_initial_state = initial_state.clone()
        cf_initial_state[0:3] += torch.tensor(delta_pos, dtype=torch.float32, device=self.engine.device)
        cf_initial_state[3:6] += torch.tensor(delta_vel, dtype=torch.float32, device=self.engine.device)

        cf_spin = actual_spin.clone()
        cf_spin += torch.tensor(delta_spin, dtype=torch.float32, device=self.engine.device)

        # 5. 반사실적 궤적 시뮬레이션
        cf_time, cf_traj = self._simulate_trajectory(cf_initial_state, cf_spin)

        # 6. Approach Angles 계산
        actual_angles = self.calculate_approach_angles(actual_traj)
        cf_angles = self.calculate_approach_angles(cf_traj)

        return {
            'actual_time': actual_time,
            'actual_traj': actual_traj,
            'cf_time': cf_time,
            'cf_traj': cf_traj,
            'initial_state': initial_state.cpu().numpy(),
            'actual_spin': actual_spin.cpu().numpy(),
            'cf_spin': cf_spin.cpu().numpy(),
            'actual_pitch_type': actual_pitch_data.get('pitch_type', 'Unknown'),
            'target_pitch_type': target_pitch_type,
            'actual_vaa': actual_angles['vaa'],
            'cf_vaa': cf_angles['vaa'],
            'actual_haa': actual_angles['haa'],
            'cf_haa': cf_angles['haa']
        }

    def calculate_tunnel_score(
        self,
        traj1: np.ndarray,
        time1: np.ndarray,
        traj2: np.ndarray,
        time2: np.ndarray
    ) -> Dict[str, float]:
        """
        두 궤적 간의 터널링 점수 계산

        Args:
            traj1: 첫 번째 궤적 [N, 6]
            time1: 첫 번째 시간 [N]
            traj2: 두 번째 궤적 [M, 6]
            time2: 두 번째 시간 [M]

        Returns:
            dict: {
                'tunnel_score': 터널 점수 (0~1),
                'distance_at_decision': Decision Point에서의 거리 (m),
                'decision_point_pos1': Decision Point에서의 궤적1 위치,
                'decision_point_pos2': Decision Point에서의 궤적2 위치
            }
        """
        # Decision Point에서의 위치 보간
        def get_position_at_time(traj, time, target_t):
            """특정 시간에서의 위치를 선형 보간으로 구함"""
            if target_t <= time[0]:
                return traj[0, :3]
            if target_t >= time[-1]:
                return traj[-1, :3]

            # 선형 보간
            idx = np.searchsorted(time, target_t)
            if idx == 0:
                return traj[0, :3]

            t0, t1 = time[idx-1], time[idx]
            p0, p1 = traj[idx-1, :3], traj[idx, :3]

            alpha = (target_t - t0) / (t1 - t0)
            return p0 + alpha * (p1 - p0)

        pos1_decision = get_position_at_time(traj1, time1, self.DECISION_TIME)
        pos2_decision = get_position_at_time(traj2, time2, self.DECISION_TIME)

        # 유클리드 거리
        distance = np.linalg.norm(pos1_decision - pos2_decision)

        # Tunnel Score: 1 / (1 + distance)
        # 거리가 0이면 1.0, 거리가 클수록 0에 가까워짐
        tunnel_score = 1.0 / (1.0 + distance)

        return {
            'tunnel_score': tunnel_score,
            'distance_at_decision': distance,
            'decision_point_pos1': pos1_decision,
            'decision_point_pos2': pos2_decision,
        }

    def visualize_tunneling(
        self,
        result: Dict,
        save_path: Optional[str] = None,
        show_decision_point: bool = True
    ):
        """
        타자 시점에서 궤적 비교 시각화

        Args:
            result: simulate_counterfactual() 결과
            save_path: 저장 경로 (None이면 표시만)
            show_decision_point: Decision Point 표시 여부
        """
        actual_traj = result['actual_traj']
        cf_traj = result['cf_traj']
        actual_time = result['actual_time']
        cf_time = result['cf_time']

        # 터널 점수 계산
        tunnel_info = self.calculate_tunnel_score(
            actual_traj, actual_time,
            cf_traj, cf_time
        )

        # 그래프 생성
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # 1. 측면 뷰 (Y-Z 평면)
        ax = axes[0]
        ax.plot(actual_traj[:, 1], actual_traj[:, 2],
                'b-', linewidth=2, label=f"Actual ({result['actual_pitch_type']})")
        ax.plot(cf_traj[:, 1], cf_traj[:, 2],
                'r--', linewidth=2, label=f"Counterfactual ({result['target_pitch_type']})")

        # Decision Point 표시
        if show_decision_point:
            pos1 = tunnel_info['decision_point_pos1']
            pos2 = tunnel_info['decision_point_pos2']
            ax.plot(pos1[1], pos1[2], 'bo', markersize=10, label='Decision Point (Actual)')
            ax.plot(pos2[1], pos2[2], 'ro', markersize=10, label='Decision Point (CF)')
            ax.plot([pos1[1], pos2[1]], [pos1[2], pos2[2]], 'k:', linewidth=1)

        ax.set_xlabel('Distance from Home Plate (m)', fontsize=12)
        ax.set_ylabel('Height (m)', fontsize=12)
        ax.set_title('Side View', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()  # 투수 → 홈플레이트

        # 스트라이크 존 표시 (홈플레이트 위치)
        strike_zone_y = 0.0  # 홈플레이트
        strike_zone_z_bot = 0.46  # 1.5 ft
        strike_zone_z_top = 1.07  # 3.5 ft
        ax.axvline(strike_zone_y, color='gray', linestyle='--', alpha=0.5)
        ax.axhspan(strike_zone_z_bot, strike_zone_z_top,
                   xmin=0, xmax=0.1, alpha=0.2, color='green')

        # 2. 타자 시점 (X-Z 평면)
        ax = axes[1]
        ax.plot(actual_traj[:, 0], actual_traj[:, 2],
                'b-', linewidth=2, label=f"Actual ({result['actual_pitch_type']})")
        ax.plot(cf_traj[:, 0], cf_traj[:, 2],
                'r--', linewidth=2, label=f"Counterfactual ({result['target_pitch_type']})")

        # Decision Point 표시
        if show_decision_point:
            pos1 = tunnel_info['decision_point_pos1']
            pos2 = tunnel_info['decision_point_pos2']
            ax.plot(pos1[0], pos1[2], 'bo', markersize=10, label='Decision Point (Actual)')
            ax.plot(pos2[0], pos2[2], 'ro', markersize=10, label='Decision Point (CF)')
            ax.plot([pos1[0], pos2[0]], [pos1[2], pos2[2]], 'k:', linewidth=1)

        ax.set_xlabel('Horizontal (m)', fontsize=12)
        ax.set_ylabel('Height (m)', fontsize=12)
        ax.set_title("Batter's View", fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 스트라이크 존 표시
        strike_zone_width = 0.43  # 17 inches
        ax.add_patch(plt.Rectangle(
            (-strike_zone_width/2, strike_zone_z_bot),
            strike_zone_width,
            strike_zone_z_top - strike_zone_z_bot,
            fill=False, edgecolor='green', linewidth=2
        ))

        # 터널 점수 및 VAA 표시
        actual_vaa = result.get('actual_vaa', 0.0)
        cf_vaa = result.get('cf_vaa', 0.0)
        actual_haa = result.get('actual_haa', 0.0)
        cf_haa = result.get('cf_haa', 0.0)

        fig.suptitle(
            f"Tunnel Score: {tunnel_info['tunnel_score']:.3f} | "
            f"Distance: {tunnel_info['distance_at_decision']:.3f}m | "
            f"VAA: {result['actual_pitch_type']}={actual_vaa:.2f}° / {result['target_pitch_type']}={cf_vaa:.2f}°",
            fontsize=13, fontweight='bold'
        )

        # VAA/HAA 상세 정보를 그래프 하단에 추가
        info_text = (
            f"Approach Angles:\n"
            f"  {result['actual_pitch_type']}: VAA={actual_vaa:.2f}°, HAA={actual_haa:.2f}°\n"
            f"  {result['target_pitch_type']}: VAA={cf_vaa:.2f}°, HAA={cf_haa:.2f}°"
        )
        fig.text(0.5, 0.01, info_text, ha='center', fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📊 시각화 저장: {save_path}")
        else:
            plt.show()

        plt.close()

        return tunnel_info


def main():
    """사용 예시 - Production Version"""
    print("=" * 80)
    print("🎯 TunnelingAnalyzer 예제 (Production Version)")
    print("=" * 80 + "\n")

    # 1. 데이터 로더로 실제 투구 데이터 가져오기
    print("📊 Statcast 데이터 로드...")
    loader = AegisDataLoader()
    with loader as loader_context:
        df = loader_context.load_data_by_year(year=2024, limit=500)

    if df.empty:
        print("❌ 데이터를 찾을 수 없습니다.")
        return

    # Fastball 투구 선택 (pitcher_id가 있는 데이터)
    fastball_df = df[df['pitch_type'] == 'FF']
    if fastball_df.empty:
        print("❌ Fastball 데이터를 찾을 수 없습니다.")
        return

    fastball_data = fastball_df.iloc[0]
    pitcher_id = int(fastball_data['pitcher']) if 'pitcher' in fastball_data else None

    print(f"   투수 ID: {pitcher_id}")
    print(f"   투구 타입: {fastball_data['pitch_type']}")
    print(f"   속도: {fastball_data['release_speed']:.1f} mph")
    print(f"   회전: {fastball_data['release_spin_rate']:.0f} RPM\n")

    # 2. TunnelingAnalyzer 초기화 (data_loader 전달)
    analyzer = TunnelingAnalyzer(data_loader=loader)
    print()

    # 3. 반사실적 시뮬레이션 (Delta Injection Method)
    print("🔬 반사실적 시뮬레이션 (Fastball → Slider)...")
    print("   Method: Delta Injection\n")

    result = analyzer.simulate_counterfactual(
        actual_pitch_data=fastball_data,
        target_pitch_type='SL',
        pitcher_id=pitcher_id
    )

    print(f"   실제 궤적: {len(result['actual_time'])} 포인트")
    print(f"   반사실적 궤적: {len(result['cf_time'])} 포인트")
    print(f"   실제 VAA: {result['actual_vaa']:.2f}°")
    print(f"   반사실적 VAA: {result['cf_vaa']:.2f}°")
    print(f"   실제 HAA: {result['actual_haa']:.2f}°")
    print(f"   반사실적 HAA: {result['cf_haa']:.2f}°\n")

    # 4. 터널 점수 계산
    print("📊 터널 점수 계산...")
    tunnel_info = analyzer.calculate_tunnel_score(
        result['actual_traj'],
        result['actual_time'],
        result['cf_traj'],
        result['cf_time']
    )

    print(f"   Tunnel Score: {tunnel_info['tunnel_score']:.3f}")
    print(f"   Decision Point 거리: {tunnel_info['distance_at_decision']:.3f}m")
    print(f"   실제 위치: {tunnel_info['decision_point_pos1']}")
    print(f"   반사실적 위치: {tunnel_info['decision_point_pos2']}\n")

    # 5. 시각화
    print("📈 시각화 생성...")
    analyzer.visualize_tunneling(
        result,
        save_path='examples/tunneling_analysis.png'
    )
    print()

    # 6. 여러 투구 타입 비교
    print("=" * 80)
    print("🔄 여러 투구 타입과 비교")
    print("=" * 80 + "\n")

    target_types = ['SI', 'FC', 'SL', 'CU', 'CH']
    scores = []

    for target_type in target_types:
        result = analyzer.simulate_counterfactual(
            fastball_data,
            target_type,
            pitcher_id=pitcher_id
        )
        tunnel_info = analyzer.calculate_tunnel_score(
            result['actual_traj'], result['actual_time'],
            result['cf_traj'], result['cf_time']
        )
        scores.append({
            'target': target_type,
            'score': tunnel_info['tunnel_score'],
            'distance': tunnel_info['distance_at_decision'],
            'vaa': result['cf_vaa'],
            'haa': result['cf_haa']
        })
        print(f"   FF → {target_type}: Score={tunnel_info['tunnel_score']:.3f}, "
              f"Distance={tunnel_info['distance_at_decision']:.3f}m, "
              f"VAA={result['cf_vaa']:.2f}°")

    print()

    # 최고 터널링 조합
    best = max(scores, key=lambda x: x['score'])
    print(f"🏆 최고 터널링 조합: FF → {best['target']} "
          f"(Score: {best['score']:.3f}, VAA: {best['vaa']:.2f}°)")

    print("\n" + "=" * 80)
    print("✅ 완료 - Production Version")
    print("=" * 80)


if __name__ == "__main__":
    main()
