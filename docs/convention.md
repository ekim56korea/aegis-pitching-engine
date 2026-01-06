# Engineering Conventions

최고 수준의 코드 품질 유지를 위한 팀 규칙입니다.

## 1. Code Style
* **Formatter:** Black (Line length 88)
* **Linter:** Flake8, Pylint
* **Type Hinting:** 모든 함수 인자 및 반환값에 Type Hint 필수 (MyPy 호환).

## 2. Git Workflow
* **Main Branch:** `main` (Production-ready code only)
* **Dev Branch:** `develop`
* **Feature Branch:** `feature/feature-name`
* **Commit Message:** [Conventional Commits](https://www.conventionalcommits.org/) 준수
    * `feat`: 새로운 기능
    * `fix`: 버그 수정
    * `docs`: 문서 수정
    * `refactor`: 코드 리팩토링

## 3. Documentation
* 모든 모듈과 클래스는 Docstring을 포함해야 함 (Google Style).
* 복잡한 수학적 로직(PINNs 등)은 반드시 주석에 LaTeX 수식 설명 첨부.
