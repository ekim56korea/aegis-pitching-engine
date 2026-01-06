"""
PitchTrajectoryPINN: Physics-Informed Neural Network for Baseball Pitch Trajectory
물리 법칙을 손실 함수에 반영한 궤적 예측 신경망
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.physics_engine import SavantPhysicsEngine


class TrajectoryNet(nn.Module):
    """
    시간과 초기 상태를 입력받아 3D 위치를 출력하는 MLP

    Architecture:
        Input: (t, initial_state) -> [batch_size, 7]
            - t: 시간 (1차원)
            - initial_state: (x0, y0, z0, vx0, vy0, vz0) (6차원)
        Hidden: 4 layers × 128 units with Tanh activation
        Output: (x, y, z) -> [batch_size, 3]
    """

    def __init__(
        self,
        input_dim: int = 7,
        hidden_dim: int = 128,
        num_layers: int = 4,
        output_dim: int = 3
    ):
        """
        신경망 초기화

        Args:
            input_dim: 입력 차원 (시간 1 + 초기 상태 6 = 7)
            hidden_dim: 은닉층 유닛 수
            num_layers: 은닉층 개수
            output_dim: 출력 차원 (x, y, z = 3)
        """
        super(TrajectoryNet, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim

        # 레이어 구성
        layers = []

        # 입력층 -> 첫 번째 은닉층
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh())

        # 은닉층들 (Tanh: 미분 가능성 확보)
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())

        # 출력층
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.network = nn.Sequential(*layers)

        # Xavier 초기화
        self._initialize_weights()

    def _initialize_weights(self):
        """Xavier/Glorot 초기화"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        t: torch.Tensor,
        initial_state: torch.Tensor
    ) -> torch.Tensor:
        """
        순전파

        Args:
            t: 시간 [batch_size, 1]
            initial_state: 초기 상태 (x0, y0, z0, vx0, vy0, vz0) [batch_size, 6]

        Returns:
            torch.Tensor: 예측된 위치 (x, y, z) [batch_size, 3]
        """
        # 입력 결합: [t, initial_state]
        x = torch.cat([t, initial_state], dim=-1)  # [batch_size, 7]

        # 신경망 통과
        position = self.network(x)  # [batch_size, 3]

        return position


class PitchTrajectoryPINN:
    """
    Physics-Informed Neural Network for Pitch Trajectory Prediction

    Features:
        - 물리 법칙을 손실 함수에 직접 반영
        - 자동 미분을 통한 속도/가속도 계산
        - 데이터 손실 + 물리 손실 결합
    """

    def __init__(
        self,
        physics_engine: SavantPhysicsEngine,
        hidden_dim: int = 128,
        num_layers: int = 4,
        device: str = 'cpu'
    ):
        """
        PINN 초기화

        Args:
            physics_engine: 물리 엔진 인스턴스
            hidden_dim: 은닉층 유닛 수
            num_layers: 은닉층 개수
            device: PyTorch 디바이스
        """
        self.device = torch.device(device)
        self.physics_engine = physics_engine

        # 신경망 생성
        self.model = TrajectoryNet(
            input_dim=7,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=3
        ).to(self.device)

        print(f"✅ PitchTrajectoryPINN 초기화")
        print(f"   모델 구조: {num_layers} layers × {hidden_dim} units")
        print(f"   디바이스: {self.device}")
        print(f"   파라미터 수: {sum(p.numel() for p in self.model.parameters()):,}")

    def compute_physics_loss(
        self,
        t: torch.Tensor,
        initial_state: torch.Tensor,
        spin_vec: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        물리 손실 계산 (핵심 메서드)

        Args:
            t: 시간 [batch_size, 1] - requires_grad=True 필요
            initial_state: 초기 상태 [batch_size, 6]
            spin_vec: 회전 벡터 [batch_size, 3]

        Returns:
            tuple: (physics_loss, diagnostics)
                - physics_loss: 물리 손실 값
                - diagnostics: 디버깅용 중간 값들
        """
        # 1. 모델 예측 (위치)
        position_pred = self.model(t, initial_state)  # [batch_size, 3]

        # 2. 속도 계산 (자동 미분: ∂position/∂t)
        # 각 위치 성분에 대해 미분
        velocity_components = []
        for i in range(3):  # x, y, z
            grad = torch.autograd.grad(
                outputs=position_pred[:, i].sum(),  # 스칼라로 만들기
                inputs=t,
                create_graph=True,  # 2차 미분을 위해 필요
                retain_graph=True
            )[0]  # [batch_size, 1]
            velocity_components.append(grad)

        velocity_pred = torch.cat(velocity_components, dim=-1)  # [batch_size, 3]

        # 3. 가속도 계산 (자동 미분: ∂velocity/∂t)
        acceleration_components = []
        for i in range(3):  # x, y, z
            grad = torch.autograd.grad(
                outputs=velocity_pred[:, i].sum(),
                inputs=t,
                create_graph=True,
                retain_graph=True
            )[0]  # [batch_size, 1]
            acceleration_components.append(grad)

        acceleration_pred = torch.cat(acceleration_components, dim=-1)  # [batch_size, 3]

        # 4. 물리 엔진으로 실제 가속도 계산
        # 상태 벡터 구성: [position_pred, velocity_pred]
        state = torch.cat([position_pred, velocity_pred], dim=-1)  # [batch_size, 6]

        # 물리 법칙에 의한 힘 계산
        forces = self.physics_engine.compute_forces(state, spin_vec)  # [batch_size, 3]

        # F = ma → a = F/m
        acceleration_real = forces / self.physics_engine.mass  # [batch_size, 3]

        # 5. 물리 손실: 예측 가속도 vs 실제 가속도
        physics_loss = torch.mean((acceleration_pred - acceleration_real) ** 2)

        # 디버깅 정보
        diagnostics = {
            'position_pred': position_pred.detach(),
            'velocity_pred': velocity_pred.detach(),
            'acceleration_pred': acceleration_pred.detach(),
            'acceleration_real': acceleration_real.detach(),
            'forces': forces.detach(),
        }

        return physics_loss, diagnostics

    def compute_data_loss(
        self,
        t: torch.Tensor,
        initial_state: torch.Tensor,
        target_position: torch.Tensor
    ) -> torch.Tensor:
        """
        데이터 손실 계산 (관측 데이터와의 오차)

        Args:
            t: 시간 [batch_size, 1]
            initial_state: 초기 상태 [batch_size, 6]
            target_position: 실제 관측 위치 [batch_size, 3]

        Returns:
            torch.Tensor: 데이터 손실 값
        """
        # 모델 예측
        position_pred = self.model(t, initial_state)  # [batch_size, 3]

        # MSE 손실
        data_loss = torch.mean((position_pred - target_position) ** 2)

        return data_loss

    def compute_total_loss(
        self,
        t_physics: torch.Tensor,
        initial_state: torch.Tensor,
        spin_vec: torch.Tensor,
        t_data: Optional[torch.Tensor] = None,
        target_position: Optional[torch.Tensor] = None,
        lambda_physics: float = 1.0,
        lambda_data: float = 1.0
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        전체 손실 계산 (물리 손실 + 데이터 손실)

        Args:
            t_physics: 물리 제약을 적용할 시간 포인트 [batch_size, 1]
            initial_state: 초기 상태 [batch_size, 6]
            spin_vec: 회전 벡터 [batch_size, 3]
            t_data: 데이터 관측 시간 [batch_size, 1] (optional)
            target_position: 관측 위치 [batch_size, 3] (optional)
            lambda_physics: 물리 손실 가중치
            lambda_data: 데이터 손실 가중치

        Returns:
            tuple: (total_loss, loss_dict)
        """
        # 물리 손실
        physics_loss, diagnostics = self.compute_physics_loss(
            t_physics, initial_state, spin_vec
        )

        # 데이터 손실
        if t_data is not None and target_position is not None:
            data_loss = self.compute_data_loss(
                t_data, initial_state, target_position
            )
        else:
            data_loss = torch.tensor(0.0, device=self.device)

        # 전체 손실
        total_loss = lambda_physics * physics_loss + lambda_data * data_loss

        loss_dict = {
            'total': total_loss.item(),
            'physics': physics_loss.item(),
            'data': data_loss.item(),
        }

        return total_loss, loss_dict

    def predict_trajectory(
        self,
        t_points: torch.Tensor,
        initial_state: torch.Tensor
    ) -> torch.Tensor:
        """
        전체 궤적 예측

        Args:
            t_points: 시간 포인트들 [num_points, 1]
            initial_state: 초기 상태 [1, 6] 또는 [batch_size, 6]

        Returns:
            torch.Tensor: 예측된 궤적 [num_points, 3] 또는 [batch_size, num_points, 3]
        """
        self.model.eval()

        with torch.no_grad():
            # initial_state를 t_points 크기에 맞춰 복사
            if initial_state.shape[0] == 1:
                initial_state_expanded = initial_state.repeat(t_points.shape[0], 1)
            else:
                initial_state_expanded = initial_state

            positions = self.model(t_points, initial_state_expanded)

        return positions

    def save_model(self, path: str):
        """모델 저장"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'hidden_dim': self.model.hidden_dim,
                'num_layers': self.model.num_layers,
            }
        }, path)
        print(f"💾 모델 저장: {path}")

    def load_model(self, path: str):
        """모델 로드"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"📂 모델 로드: {path}")


def main():
    """사용 예시 및 테스트"""
    print("=" * 80)
    print("🚀 PitchTrajectoryPINN 테스트")
    print("=" * 80 + "\n")

    # 1. 물리 엔진 초기화
    engine = SavantPhysicsEngine(
        temperature_f=70.0,
        pressure_hg=29.92,
        humidity_percent=50.0,
        elevation_ft=0.0
    )
    print()

    # 2. PINN 초기화
    pinn = PitchTrajectoryPINN(
        physics_engine=engine,
        hidden_dim=128,
        num_layers=4,
        device='cpu'
    )
    print()

    # 3. 샘플 데이터 생성
    print("=" * 80)
    print("📊 샘플 데이터 생성")
    print("=" * 80 + "\n")

    batch_size = 4

    # 시간 (requires_grad=True: 자동 미분을 위해 필요)
    t = torch.linspace(0.0, 0.5, batch_size).unsqueeze(1).requires_grad_(True)

    # 초기 상태 (릴리즈 포인트)
    initial_state = torch.tensor([
        [0.0, 18.44, 1.83, 0.0, -42.5, 0.0],  # Fastball
        [0.0, 18.44, 1.83, 0.0, -38.0, 0.0],  # Slider
        [0.0, 18.44, 1.83, 0.0, -35.0, 0.0],  # Curveball
        [0.0, 18.44, 1.83, 0.0, -40.0, 0.0],  # Changeup
    ], dtype=torch.float32)

    # 회전 벡터
    spin_vec = torch.tensor([
        [251.3, 0.0, 0.0],      # Fastball: 2400 RPM backspin
        [150.0, 0.0, 220.0],    # Slider
        [0.0, 0.0, -280.0],     # Curveball
        [100.0, 0.0, 0.0],      # Changeup
    ], dtype=torch.float32)

    print(f"배치 크기: {batch_size}")
    print(f"시간 범위: {t[0].item():.2f}s ~ {t[-1].item():.2f}s")
    print()

    # 4. 물리 손실 계산 테스트
    print("=" * 80)
    print("🔬 물리 손실 계산")
    print("=" * 80 + "\n")

    physics_loss, diagnostics = pinn.compute_physics_loss(t, initial_state, spin_vec)

    print(f"물리 손실: {physics_loss.item():.6f}\n")

    print("예측된 값들 (첫 번째 샘플):")
    print(f"  위치: {diagnostics['position_pred'][0].numpy()}")
    print(f"  속도: {diagnostics['velocity_pred'][0].numpy()}")
    print(f"  가속도(예측): {diagnostics['acceleration_pred'][0].numpy()}")
    print(f"  가속도(물리): {diagnostics['acceleration_real'][0].numpy()}")
    print(f"  힘: {diagnostics['forces'][0].numpy()}")
    print()

    # 5. 데이터 손실 계산 테스트
    print("=" * 80)
    print("📍 데이터 손실 계산")
    print("=" * 80 + "\n")

    # 가상의 관측 데이터 (홈플레이트 위치)
    t_final = torch.tensor([[0.5]], dtype=torch.float32)
    target_position = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)  # 홈플레이트

    data_loss = pinn.compute_data_loss(
        t_final,
        initial_state[0:1],  # 첫 번째 샘플만
        target_position
    )

    print(f"데이터 손실: {data_loss.item():.6f}\n")

    # 6. 전체 손실 계산
    print("=" * 80)
    print("📊 전체 손실 계산")
    print("=" * 80 + "\n")

    # 데이터 손실용 배치 생성 (batch_size에 맞춤)
    t_data_batch = torch.full((batch_size, 1), 0.5, dtype=torch.float32)
    target_position_batch = torch.tensor([
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
    ], dtype=torch.float32)

    total_loss, loss_dict = pinn.compute_total_loss(
        t_physics=t,
        initial_state=initial_state,
        spin_vec=spin_vec,
        t_data=t_data_batch,
        target_position=target_position_batch,
        lambda_physics=1.0,
        lambda_data=10.0
    )

    print("손실 분해:")
    print(f"  전체 손실: {loss_dict['total']:.6f}")
    print(f"  물리 손실: {loss_dict['physics']:.6f}")
    print(f"  데이터 손실: {loss_dict['data']:.6f}")
    print()

    # 7. 궤적 예측
    print("=" * 80)
    print("🎯 궤적 예측")
    print("=" * 80 + "\n")

    t_trajectory = torch.linspace(0.0, 0.5, 11).unsqueeze(1)
    trajectory = pinn.predict_trajectory(t_trajectory, initial_state[0:1])

    print("시간별 예측 위치 (Fastball):")
    print("시간(s)    x(m)      y(m)      z(m)")
    print("-" * 45)
    for i, (time, pos) in enumerate(zip(t_trajectory, trajectory)):
        print(f"{time.item():.2f}     {pos[0].item():+7.3f}  {pos[1].item():+7.3f}  {pos[2].item():+7.3f}")

    print("\n" + "=" * 80)
    print("✅ 테스트 완료")
    print("=" * 80)


if __name__ == "__main__":
    main()
