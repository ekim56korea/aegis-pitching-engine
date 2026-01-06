"""
PINN 학습 예제: 단순 궤적 학습
물리 손실과 데이터 손실을 결합하여 PINN 학습
"""

import torch
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')  # GUI 없이 사용
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.physics_engine import SavantPhysicsEngine
from src.physics_engine.pinn import PitchTrajectoryPINN


def generate_training_data(
    engine: SavantPhysicsEngine,
    n_samples: int = 10
) -> tuple:
    """
    학습용 데이터 생성 (물리 시뮬레이션 사용)

    Args:
        engine: 물리 엔진
        n_samples: 샘플 수

    Returns:
        tuple: (t_points, positions, initial_state, spin_vec)
    """
    # 초기 조건
    initial_state = torch.tensor([
        0.0,      # x
        18.44,    # y: 투수판
        1.83,     # z: 릴리즈 높이
        0.0,      # vx
        -42.5,    # vy: 홈플레이트 방향
        0.0       # vz
    ], dtype=torch.float32)

    # Backspin
    spin_vec = torch.tensor([251.3, 0.0, 0.0], dtype=torch.float32)

    # 시간 포인트
    t_points = torch.linspace(0.0, 0.5, n_samples).unsqueeze(1)

    # 간단한 Euler 적분으로 실제 궤적 생성
    dt = 0.5 / (n_samples - 1)
    positions = []
    current_state = initial_state.clone()

    for t in t_points:
        positions.append(current_state[:3].clone())

        # 힘 계산
        state_6d = current_state.unsqueeze(0)
        spin_6d = spin_vec.unsqueeze(0)
        forces = engine.compute_forces(state_6d, spin_6d).squeeze(0)
        accel = forces / engine.mass

        # 상태 업데이트
        current_state[3:6] += accel * dt
        current_state[0:3] += current_state[3:6] * dt

    positions = torch.stack(positions)

    return t_points, positions, initial_state, spin_vec


def train_pinn(
    pinn: PitchTrajectoryPINN,
    t_physics: torch.Tensor,
    t_data: torch.Tensor,
    target_positions: torch.Tensor,
    initial_state: torch.Tensor,
    spin_vec: torch.Tensor,
    n_epochs: int = 1000,
    lr: float = 1e-3,
    lambda_physics: float = 1.0,
    lambda_data: float = 10.0
):
    """
    PINN 학습

    Args:
        pinn: PINN 인스턴스
        t_physics: 물리 제약 시간 포인트
        t_data: 데이터 관측 시간 포인트
        target_positions: 타겟 위치
        initial_state: 초기 상태
        spin_vec: 회전 벡터
        n_epochs: 에폭 수
        lr: 학습률
        lambda_physics: 물리 손실 가중치
        lambda_data: 데이터 손실 가중치
    """
    optimizer = optim.Adam(pinn.model.parameters(), lr=lr)

    history = {
        'total_loss': [],
        'physics_loss': [],
        'data_loss': []
    }

    print(f"🚀 학습 시작 (에폭: {n_epochs}, 학습률: {lr})")
    print(f"   λ_physics: {lambda_physics}, λ_data: {lambda_data}")
    print("-" * 80)

    # 배치 크기에 맞게 확장
    batch_size = t_physics.shape[0]
    initial_state_batch = initial_state.unsqueeze(0).repeat(batch_size, 1)
    spin_vec_batch = spin_vec.unsqueeze(0).repeat(batch_size, 1)

    for epoch in range(n_epochs):
        pinn.model.train()
        optimizer.zero_grad()

        # 손실 계산
        total_loss, loss_dict = pinn.compute_total_loss(
            t_physics=t_physics,
            initial_state=initial_state_batch,
            spin_vec=spin_vec_batch,
            t_data=t_data,
            target_position=target_positions,
            lambda_physics=lambda_physics,
            lambda_data=lambda_data
        )

        # 역전파
        total_loss.backward()
        optimizer.step()

        # 기록
        history['total_loss'].append(loss_dict['total'])
        history['physics_loss'].append(loss_dict['physics'])
        history['data_loss'].append(loss_dict['data'])

        # 로그 출력
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1:4d} | "
                  f"Total: {loss_dict['total']:.6f} | "
                  f"Physics: {loss_dict['physics']:.6f} | "
                  f"Data: {loss_dict['data']:.6f}")

    print("-" * 80)
    print("✅ 학습 완료\n")

    return history


def plot_results(
    pinn: PitchTrajectoryPINN,
    t_points: torch.Tensor,
    true_positions: torch.Tensor,
    initial_state: torch.Tensor,
    history: dict,
    save_path: str = 'examples/pinn_results.png'
):
    """결과 시각화"""
    # 예측
    pinn.model.eval()
    with torch.no_grad():
        pred_positions = pinn.predict_trajectory(t_points, initial_state.unsqueeze(0))

    # 그래프 생성
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 손실 곡선
    ax = axes[0, 0]
    ax.plot(history['total_loss'], label='Total Loss', linewidth=2)
    ax.plot(history['physics_loss'], label='Physics Loss', alpha=0.7)
    ax.plot(history['data_loss'], label='Data Loss', alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # 2. Y-Z 평면 (측면 뷰)
    ax = axes[0, 1]
    ax.plot(true_positions[:, 1].numpy(), true_positions[:, 2].numpy(),
            'o-', label='True', markersize=8, linewidth=2)
    ax.plot(pred_positions[:, 1].numpy(), pred_positions[:, 2].numpy(),
            's--', label='Predicted', markersize=6, linewidth=2)
    ax.set_xlabel('Y (m) - Distance to Home Plate')
    ax.set_ylabel('Z (m) - Height')
    ax.set_title('Trajectory (Side View)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()  # 투수 → 홈플레이트 방향

    # 3. X-Y 평면 (위에서 본 뷰)
    ax = axes[1, 0]
    ax.plot(true_positions[:, 0].numpy(), true_positions[:, 1].numpy(),
            'o-', label='True', markersize=8, linewidth=2)
    ax.plot(pred_positions[:, 0].numpy(), pred_positions[:, 1].numpy(),
            's--', label='Predicted', markersize=6, linewidth=2)
    ax.set_xlabel('X (m) - Horizontal')
    ax.set_ylabel('Y (m) - Distance')
    ax.set_title('Trajectory (Top View)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()  # 투수 → 홈플레이트 방향

    # 4. 오차 분석
    ax = axes[1, 1]
    error = torch.norm(pred_positions - true_positions, dim=1).numpy()
    ax.plot(t_points.squeeze().detach().numpy(), error, 'o-', linewidth=2, markersize=6)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Position Error (m)')
    ax.set_title('Prediction Error Over Time')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"📊 결과 그래프 저장: {save_path}")
    plt.close()


def main():
    print("=" * 80)
    print("🎓 PINN 학습 예제")
    print("=" * 80 + "\n")

    # 1. 물리 엔진 초기화
    engine = SavantPhysicsEngine(
        temperature_f=70.0,
        pressure_hg=29.92,
        humidity_percent=50.0,
        elevation_ft=0.0
    )
    print()

    # 2. 학습 데이터 생성
    print("📊 학습 데이터 생성...")
    t_points, true_positions, initial_state, spin_vec = generate_training_data(
        engine, n_samples=10
    )
    print(f"   시간 포인트: {len(t_points)}개")
    print(f"   시간 범위: {t_points[0].item():.2f}s ~ {t_points[-1].item():.2f}s")
    print()

    # 3. PINN 초기화
    pinn = PitchTrajectoryPINN(
        physics_engine=engine,
        hidden_dim=128,
        num_layers=4,
        device='cpu'
    )
    print()

    # 4. 학습 전 예측
    print("🔍 학습 전 예측...")
    pinn.model.eval()
    with torch.no_grad():
        pred_before = pinn.predict_trajectory(t_points, initial_state.unsqueeze(0))
        error_before = torch.mean(torch.norm(pred_before - true_positions, dim=1)).item()
    print(f"   평균 오차: {error_before:.4f} m\n")

    # 5. 학습
    t_physics = t_points.requires_grad_(True)

    history = train_pinn(
        pinn=pinn,
        t_physics=t_physics,
        t_data=t_points,
        target_positions=true_positions,
        initial_state=initial_state,
        spin_vec=spin_vec,
        n_epochs=1000,
        lr=1e-3,
        lambda_physics=1.0,
        lambda_data=10.0
    )

    # 6. 학습 후 예측
    print("🎯 학습 후 예측...")
    pinn.model.eval()
    with torch.no_grad():
        pred_after = pinn.predict_trajectory(t_points, initial_state.unsqueeze(0))
        error_after = torch.mean(torch.norm(pred_after - true_positions, dim=1)).item()
    print(f"   평균 오차: {error_after:.4f} m")
    print(f"   개선율: {(1 - error_after/error_before)*100:.1f}%\n")

    # 7. 결과 시각화
    print("📈 결과 시각화...")
    plot_results(pinn, t_points, true_positions, initial_state, history)
    print()

    # 8. 모델 저장
    save_path = "examples/pinn_model.pt"
    pinn.save_model(save_path)
    print()

    print("=" * 80)
    print("✅ 완료")
    print("=" * 80)


if __name__ == "__main__":
    main()
