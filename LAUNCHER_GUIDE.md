# 🚀 Dashboard Launcher Scripts

이 디렉토리에는 Aegis Strategy Room 대시보드를 실행하기 위한 여러 스크립트가 포함되어 있습니다.

---

## 📁 사용 가능한 스크립트

### 1. `launch_dashboard.sh` ⭐ 권장

**용도**: 일반적인 상황에서 대시보드 실행

**특징**:

- ✅ 자동 의존성 확인
- ✅ IPv4/IPv6 호환성 최적화 (`--server.address=0.0.0.0`)
- ✅ 여러 접속 URL 표시
- ✅ CORS 및 XSRF 보호 비활성화 (로컬 사용)

**사용법**:

```bash
./launch_dashboard.sh
```

**접속**:

- Primary: `http://localhost:8501`
- Recommended: `http://127.0.0.1:8501`
- Network: `http://[Local-IP]:8501`

---

### 2. `launch_dashboard_debug.sh` 🔧 트러블슈팅

**용도**: 연결 문제 발생 시 상세 진단

**특징**:

- ✅ 포트 8501 충돌 자동 감지
- ✅ 충돌 프로세스 종료 옵션
- ✅ 로컬 IP 자동 탐지
- ✅ 상세 로그를 `streamlit_debug.log`에 저장
- ✅ 디버그 레벨 로깅 활성화

**사용법**:

```bash
./launch_dashboard_debug.sh
```

**출력 예시**:

```
🔧 Aegis Dashboard - Troubleshooting Mode
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Step 1: Checking port 8501...
✅ Port 8501 is available

Step 2: Network information
   Local IP: 192.168.1.100

Step 3: Connection URLs
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Try these URLs in order:

   1️⃣  http://localhost:8501          (Standard)
   2️⃣  http://127.0.0.1:8501          (IPv4 direct)
   3️⃣  http://192.168.1.100:8501      (Network IP)

💡 Recommended: Use option 2 (127.0.0.1) for best compatibility
```

---

### 3. `test_dashboard_connection.sh` 🧪 연결 테스트

**용도**: 대시보드 접속 가능 여부 자동 테스트

**특징**:

- ✅ 백그라운드에서 대시보드 실행
- ✅ `localhost` 및 `127.0.0.1` 자동 테스트
- ✅ 테스트 완료 후 자동 종료
- ✅ 어떤 URL이 작동하는지 확인

**사용법**:

```bash
./test_dashboard_connection.sh
```

**출력 예시**:

```
🧪 Testing Aegis Dashboard Connectivity...

1️⃣  Starting dashboard in background...
2️⃣  Waiting for server to start (10 seconds)...
3️⃣  Testing connectivity...

✅ localhost:8501 - OK
✅ 127.0.0.1:8501 - OK

4️⃣  Recommended URL: http://127.0.0.1:8501

5️⃣  Stopping test dashboard...

✅ Test complete!
```

---

## 🎯 사용 시나리오별 가이드

### 시나리오 1: 처음 실행하는 경우

```bash
# 1. 의존성 설치
pip install -r requirements-dashboard.txt

# 2. 표준 런처 실행
./launch_dashboard.sh

# 3. 브라우저에서 접속
# http://127.0.0.1:8501
```

---

### 시나리오 2: "사이트에 연결할 수 없음" 오류

```bash
# 1. 디버그 런처 실행
./launch_dashboard_debug.sh

# 2. 출력된 URL 중 하나 선택
# 보통 http://127.0.0.1:8501이 가장 안정적

# 3. 여전히 안 되면 트러블슈팅 가이드 확인
cat docs/TROUBLESHOOTING.md
```

---

### 시나리오 3: 포트 충돌 발생

```bash
# Option A: 디버그 런처가 자동으로 처리
./launch_dashboard_debug.sh
# 프롬프트에서 'y' 입력하여 기존 프로세스 종료

# Option B: 수동으로 포트 정리
lsof -ti:8501 | xargs kill -9
./launch_dashboard.sh
```

---

### 시나리오 4: 다른 포트로 실행

```bash
# 8502 포트로 실행
streamlit run src/dashboard/app.py \
    --server.port 8502 \
    --server.address=0.0.0.0

# 접속: http://127.0.0.1:8502
```

---

## 🔍 일반적인 문제 해결

### 문제 1: "command not found: streamlit"

**해결**:

```bash
pip install streamlit plotly
```

---

### 문제 2: "Permission denied"

**해결**:

```bash
chmod +x launch_dashboard.sh
chmod +x launch_dashboard_debug.sh
chmod +x test_dashboard_connection.sh
```

---

### 문제 3: "Module not found: src.game_theory.engine"

**해결**:

```bash
# 올바른 디렉토리로 이동
cd /Users/ekim56/Desktop/aegis-pitching-engine

# 다시 실행
./launch_dashboard.sh
```

---

### 문제 4: 대시보드가 느리거나 멈춤

**해결**:

```bash
# 기존 프로세스 정리
pkill -f streamlit

# 캐시 정리 후 재실행
rm -rf ~/.streamlit/cache
./launch_dashboard.sh
```

---

## 📊 스크립트 비교

| 특징                | launch_dashboard.sh | launch_dashboard_debug.sh | test_dashboard_connection.sh |
| ------------------- | ------------------- | ------------------------- | ---------------------------- |
| **일반 사용**       | ✅ 권장             | ⚠️ 문제 시만              | 🧪 테스트용                  |
| **포트 충돌 감지**  | ❌                  | ✅                        | ❌                           |
| **자동 정리**       | ❌                  | ✅ (선택)                 | ✅ (자동)                    |
| **상세 로그**       | ❌                  | ✅                        | ❌                           |
| **로그 파일 저장**  | ❌                  | ✅                        | ❌                           |
| **연결 테스트**     | ❌                  | ❌                        | ✅                           |
| **백그라운드 실행** | ❌                  | ❌                        | ✅                           |

---

## 🚀 권장 워크플로우

```bash
# Step 1: 첫 실행 시 연결 테스트
./test_dashboard_connection.sh

# Step 2: 테스트 성공 시 일반 런처 사용
./launch_dashboard.sh

# Step 3: 문제 발생 시 디버그 런처
./launch_dashboard_debug.sh

# Step 4: 여전히 문제 시 문서 확인
open docs/TROUBLESHOOTING.md
```

---

## 🔧 고급 사용법

### 백그라운드 실행 (서버 모드)

```bash
# 백그라운드에서 실행
nohup ./launch_dashboard.sh > dashboard.log 2>&1 &

# 프로세스 ID 확인
echo $!

# 종료하려면
pkill -f streamlit
```

---

### 특정 IP 바인딩

```bash
# 로컬 IP 확인
ifconfig | grep "inet "

# 특정 IP로 바인딩
streamlit run src/dashboard/app.py \
    --server.address=192.168.1.100
```

---

### 환경 변수 설정

```bash
# 로그 레벨 변경
STREAMLIT_LOG_LEVEL=debug ./launch_dashboard.sh

# 브라우저 자동 열기 비활성화
STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    ./launch_dashboard.sh
```

---

## 📚 추가 리소스

- **트러블슈팅 가이드**: [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)
- **사용자 가이드**: [docs/dashboard_user_guide.md](docs/dashboard_user_guide.md)
- **기술 문서**: [src/dashboard/README.md](src/dashboard/README.md)
- **Streamlit 공식 문서**: https://docs.streamlit.io/

---

## ✅ 빠른 체크리스트

실행 전 확인:

- [ ] 프로젝트 루트 디렉토리에 있는가? (`pwd` 확인)
- [ ] Virtual environment 활성화되어 있는가? (`which python` 확인)
- [ ] 의존성이 설치되어 있는가? (`pip list | grep streamlit`)
- [ ] 포트 8501이 사용 가능한가? (`lsof -i :8501`)

실행 후 확인:

- [ ] 터미널에 "You can now view..." 메시지가 보이는가?
- [ ] `http://127.0.0.1:8501` 접속이 되는가?
- [ ] 대시보드 UI가 정상적으로 로드되는가?

---

**Version**: 1.0.0
**Last Updated**: 2026-01-06
**Maintainer**: Aegis Team

**문제 신고**: 위 방법으로도 해결되지 않으면 `streamlit_debug.log`와 터미널 출력을 첨부하여 문의하세요.
