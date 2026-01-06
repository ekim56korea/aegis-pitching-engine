# Physics-Informed Neural Network (PINN) 구현 완료

## 📅 구현 날짜

2026년 1월 6일

## 🎯 구현 내용

### 1. TrajectoryNet (신경망 구조)

**파일**: [src/physics_engine/pinn.py](../src/physics_engine/pinn.py)

#### Architecture

```
Input: [t, initial_state] → [batch_size, 7]
  ├─ t: 시간 (1차원)
  └─ initial_state: (x0, y0, z0, vx0, vy0, vz0) (6차원)

Hidden Layers: 4 layers × 128 units
  └─ Activation: Tanh (미분 가능성 확보)

Output: (x, y, z) → [batch_size, 3]
```

#### 특징

- **Xavier 초기화**: 안정적인 학습을 위한 가중치 초기화
- **Tanh 활성화**: ReLU 대신 매끄러운 미분을 위해 사용
- **파라미터 수**: 50,947개

### 2. Physics Loss Computation (핵심 메서드)

#### `compute_physics_loss(t, initial_state, spin_vec)`

**자동 미분 프로세스:**

```
1. 위치 예측: position = model(t, initial_state)

2. 속도 계산 (1차 미분):
   velocity = ∂position/∂t
   → torch.autograd.grad(..., create_graph=True)

3. 가속도 계산 (2차 미분):
   acceleration_pred = ∂velocity/∂t
   → torch.autograd.grad(..., create_graph=True)

4. 물리 법칙 기반 실제 가속도:
   forces = physics_engine.compute_forces(state, spin_vec)
   acceleration_real = forces / mass

5. 물리 손실:
   Loss_physics = MSE(acceleration_pred, acceleration_real)
```

**구현 디테일:**

- 각 공간 차원(x, y, z)에 대해 독립적으로 미분
- `create_graph=True`로 2차 미분 가능
- `retain_graph=True`로 다중 backward 지원

### 3. Data Loss

#### `compute_data_loss(t, initial_state, target_position)`

```python
position_pred = model(t, initial_state)
Loss_data = MSE(position_pred, target_position)
```

관측 데이터(예: 홈플레이트 위치)와의 오차 계산

### 4. Total Loss

#### `compute_total_loss(...)`

```
Loss_total = λ_physics × Loss_physics + λ_data × Loss_data
```

**가중치 조절:**

- `λ_physics = 1.0`: 물리 법칙 준수
- `λ_data = 10.0`: 관측 데이터 피팅

## 📊 학습 결과

### 테스트 조건

- **데이터**: 10개 시간 포인트 (0.0s ~ 0.5s)
- **에폭**: 1000
- **학습률**: 0.001
- **초기 조건**: 95mph Fastball with 2400 RPM backspin

### 성능

```
학습 전 평균 오차: 9.13 m
학습 후 평균 오차: 0.20 m
개선율: 97.8%
```

### 손실 변화

```
Epoch 100:  Total: 118.57, Physics: 12.98, Data: 10.56
Epoch 500:  Total: 0.01,   Physics: 0.01,  Data: 0.00
Epoch 1000: Total: 0.15,   Physics: 0.05,  Data: 0.01
```

## 🔬 주요 기술

### 1. Automatic Differentiation

PyTorch의 자동 미분을 사용하여 수치적 안정성 확보:

- 1차 미분: 속도 계산
- 2차 미분: 가속도 계산

### 2. Physics-Informed Learning

데이터가 부족해도 물리 법칙을 통해 학습:

- 운동 방정식: F = ma
- 중력, 항력, 마그누스 힘 모두 반영

### 3. Batch Processing

효율적인 학습을 위한 배치 처리:

- 다중 시간 포인트 동시 처리
- 다중 투구 샘플 동시 학습

## 📁 생성된 파일

### 1. 핵심 코드

- **[src/physics_engine/pinn.py](../src/physics_engine/pinn.py)**: PINN 클래스 구현
  - `TrajectoryNet`: MLP 신경망
  - `PitchTrajectoryPINN`: 메인 PINN 클래스

### 2. 예제 코드

- **[examples/train_pinn.py](../examples/train_pinn.py)**: 학습 예제
  - 데이터 생성
  - 학습 루프
  - 결과 시각화

### 3. 결과 파일

- **examples/pinn_results.png**: 학습 결과 그래프
- **examples/pinn_model.pt**: 저장된 모델

## 🚀 사용 방법

### 기본 사용

```python
from src.physics_engine import SavantPhysicsEngine, PitchTrajectoryPINN
import torch

# 물리 엔진 초기화
engine = SavantPhysicsEngine(
    temperature_f=70.0,
    pressure_hg=29.92,
    humidity_percent=50.0,
    elevation_ft=0.0
)

# PINN 초기화
pinn = PitchTrajectoryPINN(
    physics_engine=engine,
    hidden_dim=128,
    num_layers=4
)

# 데이터 준비
t = torch.linspace(0, 0.5, 10).unsqueeze(1).requires_grad_(True)
initial_state = torch.tensor([[0.0, 18.44, 1.83, 0.0, -42.5, 0.0]])
spin_vec = torch.tensor([[251.3, 0.0, 0.0]])

# 물리 손실 계산
physics_loss, diagnostics = pinn.compute_physics_loss(
    t, initial_state.repeat(10, 1), spin_vec.repeat(10, 1)
)

# 궤적 예측
trajectory = pinn.predict_trajectory(t, initial_state)
```

### 학습 예제 실행

```bash
source .venv/bin/activate
python examples/train_pinn.py
```

## 🎓 이론적 배경

### Physics-Informed Neural Networks (PINNs)

Raissi et al. (2019)의 PINN 방법론을 야구 궤적 예측에 적용

**핵심 아이디어:**

1. 신경망으로 해 함수를 근사
2. 자동 미분으로 편미분 방정식 표현
3. 손실 함수에 물리 제약 포함
4. 데이터와 물리 법칙을 동시에 학습

**장점:**

- 데이터 효율성: 적은 데이터로 학습 가능
- 물리적 타당성: 물리 법칙을 자동으로 만족
- 보간/외삽 성능: 관측되지 않은 영역도 예측 가능

## 📈 향후 개선 방향

### 1. 고급 최적화

- [ ] Learning rate scheduling
- [ ] 적응적 가중치 조절 (λ_physics, λ_data)
- [ ] L-BFGS 옵티마이저 사용

### 2. 모델 확장

- [ ] 스핀 변화를 고려한 동적 모델
- [ ] 바람 효과 반영
- [ ] 투구 타입별 특화 모델

### 3. 대규모 학습

- [ ] Statcast 전체 데이터 학습
- [ ] Transfer learning
- [ ] GPU 가속 지원

### 4. 평가

- [ ] 실제 MLB 데이터와 비교
- [ ] 물리 시뮬레이터와 정확도 비교
- [ ] Cross-validation

## 📚 참고 문헌

1. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations." _Journal of Computational Physics_, 378, 686-707.

2. Nathan, A. M. (2008). "The effect of spin on the flight of a baseball." _American Journal of Physics_, 76(2), 119-124.

3. Jiménez, J. (2018). "Automatic differentiation for the numerical evaluation of derivatives." _Journal of Computational and Applied Mathematics_, 334, 78-93.

## ✅ 체크리스트

- [x] TrajectoryNet 구현 (4층 × 128 유닛, Tanh)
- [x] 자동 미분으로 속도/가속도 계산
- [x] Physics loss 구현
- [x] Data loss 구현
- [x] Batch 처리 지원
- [x] 학습 예제 작성
- [x] 결과 시각화
- [x] 모델 저장/로드 기능
- [x] Type hints 및 Docstring
- [x] 테스트 코드

---

**작성자**: Aegis Physics Engine Team
**버전**: 1.0
**최종 수정**: 2026년 1월 6일
