# System Architecture

## 1. High-Level Design
본 시스템은 **Data Lakehouse -> Physics Engine -> Strategy Engine -> User Interface**의 파이프라인으로 구성됩니다.

### Diagram
```mermaid
graph TD
    A[Raw Data (Statcast/DuckDB)] --> B(Data Pipeline / ETL)
    B --> C{Physics Engine (PINNs)}
    C -->|Simulated Trajectories| D[Counterfactual Generator]
    D --> E{Strategy Engine (CFR)}
    E -->|Nash Equilibrium| F[API Gateway (FastAPI)]
    F --> G[Frontend (React Three Fiber)]
```

## 2. Component Details

### A. Data Fabric (Src: `data_pipeline`)
* **Storage:** DuckDB (OLAP 최적화), ChromaDB (Vector Search).
* **Processing:** Airflow/Dagster 기반의 파이프라인 관리.
* **RAG:** LangChain을 활용하여 스카우팅 리포트의 텍스트 임베딩 저장.

### B. Physics Engine (Src: `physics_engine`)
* **Core:** Physics-Informed Neural Networks (PINNs).
* **Loss Function:**
  45644 Loss = Loss_{MSE} + \lambda \cdot Loss_{Physics} 45644
  * 물리 법칙(Magnus Effect, Drag Force)을 위배할 경우 페널티 부여.

### C. Strategy Engine (Src: `game_theory`)
* **Algorithm:** Deep Counterfactual Regret Minimization (Deep CFR).
* **Objective:** 투수-타자 간의 Zero-Sum Game에서 내쉬 균형 도출.

### D. Visualization (Src: `visualization`)
* **Engine:** WebGL (Three.js) via React-Three-Fiber.
* **Feature:** Tunneling Point 시각화, Ghost Ball Overlay.
