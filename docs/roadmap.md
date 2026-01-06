# Development Roadmap

## Phase 1: Foundation (Months 1-3)
- [ ] DuckDB 기반 Statcast 데이터 파이프라인 구축
- [ ] Vector DB (Chroma) 연동 및 RAG 프로토타입
- [ ] 기본 물리 방정식(Magnus Effect) Python 구현

## Phase 2: Physics Core (Months 4-8)
- [ ] PINNs 모델 아키텍처 설계 (PyTorch)
- [ ] 가상 궤적 생성 시뮬레이터 개발
- [ ] 실제 투구 vs 시뮬레이션 오차 검증

## Phase 3: Strategy Engine (Months 9-14)
- [ ] Counterfactual Regret Minimization 알고리즘 이식
- [ ] 투구 시퀀스 최적화 로직 구현
- [ ] "What-if" 시나리오 테스트

## Phase 4: Production (Months 15+)
- [ ] FastAPI 백엔드 구축 및 Redis 캐싱 적용
- [ ] TensorRT 모델 최적화
- [ ] React-Three-Fiber 프론트엔드 연동
