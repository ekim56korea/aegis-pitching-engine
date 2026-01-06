# Aegis Pitching Engine: Physics-Informed Counterfactual Sequencing

![Status](https://img.shields.io/badge/status-active-success)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)

## Project Overview

**Aegis Pitching Engine**은 MLB 투구 전략을 위한 차세대 AI 시스템입니다. 물리 기반 신경망(PINNs)과 인과 추론(Causal Inference)을 결합하여, 단순한 예측을 넘어 **"최적의 의사결정(Prescriptive Analytics)"**을 제공합니다.

## Quick Start

```bash
# 1. 가상환경 설정
python3 -m venv .venv
source .venv/bin/activate

# 2. 필수 패키지 설치
pip install duckdb pandas torch

# 3. 데이터 로더 테스트
PYTHONPATH=. python src/data_pipeline/data_loader.py

# 4. 물리 엔진 테스트
PYTHONPATH=. python src/physics_engine/savant_physics.py

# 5. 통합 데모 실행
PYTHONPATH=. python examples/statcast_physics_demo.py
```

## Core Components

### 1. 📊 Data Pipeline (`src/data_pipeline/`)

- **AegisDataLoader**: MLB Statcast 데이터 로딩 및 관리
- DuckDB 기반 고속 쿼리
- 투수별/연도별 데이터 조회
- 스키마 자동 검증

### 2. ⚙️ Physics Engine (`src/physics_engine/`)

- **SavantPhysicsEngine**: Alan Nathan Model 기반 물리 시뮬레이션
- 환경 변수 기반 공기 밀도 계산 (온도, 기압, 습도, 고도)
- Spin saturation을 고려한 고급 공기역학
- PyTorch 기반 배치 처리 지원

### 3. 🧠 Machine Learning (`src/physics_engine/`)

- **PitchTrajectoryPINN**: Physics-Informed Neural Networks
- 자동 미분을 통한 물리 법칙 준수
- 97.8% 오차 감소 (9.13m → 0.20m)
- 4-layer MLP, 50,947 parameters

### 4. 🎮 Game Theory (`src/game_theory/`)

- **TunnelingAnalyzer**: 투구 터널링 효과 분석 (Production Version)
- Delta Injection 방식의 반사실적 시뮬레이션
- VAA/HAA 계산 (Vertical/Horizontal Approach Angles)
- 투수별 구종별 평균 DNA 추출
- Decision Point (0.167s) 기반 터널 점수 계산

### 5. 🔬 Configuration (`src/common/`)

- 프로젝트 전역 설정 관리
- 물리 상수 및 필수 컬럼 정의
- 경로 관리

## Key Features

### 🎯 High-Precision Physics

- **Alan Nathan Model**: 회전-속도 상호작용을 고려한 공기역학
- **Dynamic Air Density**: 실시간 환경 조건 반영
- **Magnus Force**: 정확한 회전에 의한 궤적 변화 계산

### 📈 Real MLB Data Integration

- 780만+ 투구 데이터 (Statcast)
- 24개 필수 피처 자동 추출
- 투수/타자/시즌별 세분화된 분석

### 🚀 Production-Ready

- Type hints 완전 지원
- 배치 처리 최적화
- 예외 처리 및 로깅
- Context manager 지원

## Example Usage

### Basic Physics Simulation

```python
from src.data_pipeline import AegisDataLoader
from src.physics_engine import SavantPhysicsEngine

# 데이터 로드
with AegisDataLoader() as loader:
    df = loader.load_data_by_year(year=2024, limit=100)

# 물리 엔진 초기화
engine = SavantPhysicsEngine(
    temperature_f=70.0,
    pressure_hg=29.92,
    humidity_percent=50.0,
    elevation_ft=0.0
)

# 힘 계산
state = torch.tensor([0, 18.44, 1.83, 0, -42.5, 0])  # 위치 + 속도
spin = torch.tensor([251.3, 0, 0])  # 2400 RPM backspin
forces = engine.compute_forces(state, spin)
```

### Tunneling Analysis (Production)

```python
from src.game_theory import TunnelingAnalyzer

# Initialize
analyzer = TunnelingAnalyzer()

# Load fastball data
with AegisDataLoader() as loader:
    df = loader.load_data_by_year(year=2024, limit=100)
fastball = df[df['pitch_type'] == 'FF'].iloc[0]

# Analyze tunneling effect (FF → SL)
result = analyzer.simulate_counterfactual(
    actual_pitch_data=fastball,
    target_pitch_type='SL',
    pitcher_id=int(fastball['pitcher'])
)

# Visualize with VAA/HAA
analyzer.visualize_tunneling(result, save_path='tunneling.png')

# Results:
# - Tunnel Score: 0.812
# - Decision Point Distance: 0.232m
# - VAA: FF=-8.62° / SL=-8.88°
```

## Key Documentation

- [Architecture](./docs/architecture.md): 시스템 설계 및 데이터 흐름
- [Roadmap](./docs/roadmap.md): 개발 일정 및 마일스톤
- [Conventions](./docs/convention.md): 코딩 및 협업 규칙
- [Physics Verification](./docs/verification_report.md): 물리 엔진 검증 결과
- [PINN Implementation](./docs/pinn_implementation.md): Neural network 구현
- [Tunneling Analysis (Production)](./docs/tunneling_production.md): 터널링 분석 최종 버전
- [Tunneling Quick Reference](./docs/tunneling_quickref.md): 빠른 사용 가이드

## Project Status

### ✅ Completed Milestones

1. **Data Pipeline**: DuckDB integration with 7.8M+ pitch records
2. **Physics Engine**: Alan Nathan Model with environment-dependent aerodynamics
3. **Coordinate Verification**: All drag force tests passed
4. **PINN Training**: 97.8% error reduction achieved
5. **Tunneling Analyzer**: Production-ready with Delta Injection method
6. **VAA/HAA Metrics**: Advanced approach angle calculations
7. **Real Data Validation**: Tested with Pitcher 621111 (15,419 pitches)

### 🎯 Key Results

- **Best Tunneling**: FF → SI (Score: 0.914, Distance: 0.095m)
- **Physics Accuracy**: VAA within MLB standard range (-8° to -9°)
- **Performance**: <3s for full multi-pitch analysis

## License

MIT License
