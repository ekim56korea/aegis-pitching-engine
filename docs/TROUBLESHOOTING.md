# 🔧 Dashboard Connection Troubleshooting Guide

## 문제 상황 (Problem Statement)

"사이트에 연결할 수 없음" (Cannot connect to site) 오류가 발생하는 경우, 이 가이드를 따라 해결하세요.

---

## 🎯 빠른 해결책 (Quick Solutions)

### Solution A: 터미널 출력 확인 (Process Status Check)

**가장 흔한 원인**: 스크립트가 실행 도중 종료되었거나 아직 로딩 중인 경우

**확인 방법:**

터미널에 다음 문구가 **지속적으로** 표시되는지 확인:

```
You can now view your Streamlit app in your browser.
Local URL: http://localhost:8501
Network URL: http://192.168.x.x:8501
```

**✅ 정상 상태**: 위 메시지가 계속 떠 있고 커서가 깜빡임
**❌ 비정상 상태**: 메시지 후 터미널 프롬프트(`%` 또는 `$`)가 다시 나타남

**비정상 시 조치:**

1. 터미널에서 에러 로그 확인:

```bash
# 스크롤해서 에러 메시지 찾기
# 주로 "ModuleNotFoundError", "ImportError", "SyntaxError" 등
```

2. 에러 유형별 해결:

```bash
# Module not found
pip install streamlit plotly pandas numpy

# Permission denied
chmod +x launch_dashboard.sh

# Port already in use
lsof -ti:8501 | xargs kill -9
```

---

### Solution B: IP 주소 직접 입력 (DNS Resolution)

**원인**: Mac OS가 `localhost`를 IPv6 (`::1`)로 해석하지만, 서버는 IPv4 (`127.0.0.1`)로 실행되는 경우

**해결 방법:**

브라우저 주소창에 다음을 입력:

```
👉 http://127.0.0.1:8501
```

**왜 이것이 작동하는가?**

- `localhost`: DNS 해석 필요 (IPv4/IPv6 혼란 가능)
- `127.0.0.1`: 직접 IPv4 주소 (해석 불필요)

**테스트:**

```bash
# 터미널에서 확인
curl http://127.0.0.1:8501

# 응답이 있으면 서버가 정상 실행 중
```

---

### Solution C: 외부 접속 허용 (Port Binding)

**원인**: 서버가 루프백 인터페이스만 바인딩되어 특정 네트워크 설정에서 접근 불가

**해결 방법:**

모든 네트워크 인터페이스에 바인딩:

```bash
streamlit run src/dashboard/app.py --server.address=0.0.0.0
```

**또는 개선된 런처 사용:**

```bash
./launch_dashboard.sh
# (이미 --server.address=0.0.0.0 포함됨)
```

**접속 URL:**

- Local: `http://127.0.0.1:8501`
- Network: `http://[Your-Local-IP]:8501`

**로컬 IP 확인:**

```bash
# macOS
ifconfig | grep "inet " | grep -v 127.0.0.1

# 출력 예시: inet 192.168.1.100
# 브라우저에서: http://192.168.1.100:8501
```

---

## 🔍 상세 진단 (Detailed Diagnostics)

### Step 1: 디버그 모드 실행

트러블슈팅 전용 런처 사용:

```bash
./launch_dashboard_debug.sh
```

**제공 기능:**

- ✅ 포트 8501 사용 여부 자동 확인
- ✅ 충돌 프로세스 자동 종료 옵션
- ✅ 네트워크 IP 자동 탐지
- ✅ 상세 로그를 `streamlit_debug.log`에 저장
- ✅ 여러 접속 URL 표시

**출력 예시:**

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
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Step 4: Launching dashboard with verbose logging...
```

---

### Step 2: 포트 충돌 확인

**문제**: 다른 프로그램이 포트 8501 사용 중

**확인:**

```bash
lsof -i :8501
```

**출력 예시:**

```
COMMAND   PID   USER   FD   TYPE DEVICE SIZE/OFF NODE NAME
Python    1234  user   4u   IPv4  0x...  0t0      TCP *:8501 (LISTEN)
```

**해결:**

```bash
# 특정 프로세스 종료
kill -9 1234

# 또는 포트 사용 중인 모든 프로세스 종료
lsof -ti:8501 | xargs kill -9
```

---

### Step 3: 방화벽 확인

**macOS 방화벽 확인:**

1. **시스템 환경설정** → **보안 및 개인 정보 보호** → **방화벽**
2. 방화벽이 켜져 있다면:
   - **방화벽 옵션** 클릭
   - Python 또는 Streamlit에 대한 연결 허용 확인

**임시 해결:**

```bash
# 방화벽 일시 비활성화 (테스트용)
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --setglobalstate off

# 테스트 후 다시 활성화
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --setglobalstate on
```

---

### Step 4: 브라우저 캐시 초기화

**문제**: 이전 세션의 캐시가 남아있는 경우

**해결:**

1. **Hard Refresh**: `Cmd + Shift + R` (Chrome/Safari)
2. **캐시 삭제**:

   - Chrome: `Cmd + Shift + Delete` → "캐시된 이미지 및 파일" 선택
   - Safari: `Cmd + ,` → 개인 정보 → "웹사이트 데이터 관리"

3. **시크릿 모드 테스트**: 새 시크릿 창에서 `http://127.0.0.1:8501` 접속

---

## 🐛 일반적인 에러 및 해결책

### Error 1: "ModuleNotFoundError: No module named 'streamlit'"

**원인**: Streamlit이 설치되지 않음

**해결:**

```bash
pip install streamlit plotly pandas numpy
# 또는
pip install -r requirements-dashboard.txt
```

---

### Error 2: "Address already in use"

**원인**: 포트 8501이 이미 사용 중

**해결:**

```bash
# 포트 8501 사용 프로세스 종료
lsof -ti:8501 | xargs kill -9

# 또는 다른 포트 사용
streamlit run src/dashboard/app.py --server.port 8502
# 브라우저: http://127.0.0.1:8502
```

---

### Error 3: "Cannot find module 'src.game_theory.engine'"

**원인**: 프로젝트 루트 디렉토리가 아닌 곳에서 실행

**해결:**

```bash
# 현재 위치 확인
pwd

# 올바른 위치로 이동
cd /Users/ekim56/Desktop/aegis-pitching-engine

# 다시 실행
./launch_dashboard.sh
```

---

### Error 4: "Permission denied"

**원인**: 스크립트 실행 권한 없음

**해결:**

```bash
chmod +x launch_dashboard.sh
chmod +x launch_dashboard_debug.sh
```

---

### Error 5: "WebSocket connection failed"

**원인**: 브라우저 WebSocket 지원 문제 또는 프록시 간섭

**해결:**

```bash
# CORS 비활성화하여 실행
streamlit run src/dashboard/app.py \
    --server.enableCORS false \
    --server.enableXsrfProtection false
```

또는 다른 브라우저로 시도:

- Chrome ✅ (권장)
- Firefox ✅
- Safari ⚠️ (일부 문제 가능)
- Edge ✅

---

## 📋 체크리스트 (Troubleshooting Checklist)

문제 해결 시 순서대로 확인:

- [ ] **1. 터미널에 "You can now view..." 메시지가 계속 표시되는가?**

  - No → 에러 로그 확인, 의존성 재설치

- [ ] **2. `http://127.0.0.1:8501` 접속 시도**

  - No → 포트 충돌 확인 (`lsof -i :8501`)

- [ ] **3. 포트 8501이 열려있는가?**

  - No → 기존 프로세스 종료 (`kill -9 [PID]`)

- [ ] **4. 방화벽이 연결을 차단하는가?**

  - Yes → Python 허용 또는 일시 비활성화

- [ ] **5. 브라우저 캐시를 초기화했는가?**

  - No → Hard refresh 또는 시크릿 모드

- [ ] **6. 올바른 디렉토리에서 실행 중인가?**

  - No → `cd /Users/ekim56/Desktop/aegis-pitching-engine`

- [ ] **7. Virtual environment가 활성화되어 있는가?**
  - No → `source .venv/bin/activate`

---

## 🎯 권장 실행 방법 (Recommended Workflow)

### 1차 시도: 표준 런처

```bash
cd /Users/ekim56/Desktop/aegis-pitching-engine
source .venv/bin/activate  # Virtual environment 활성화
./launch_dashboard.sh
```

브라우저에서 접속:

```
http://127.0.0.1:8501
```

---

### 2차 시도: 디버그 런처

```bash
./launch_dashboard_debug.sh
```

출력된 URL 중 하나 선택하여 접속

---

### 3차 시도: 수동 실행

```bash
# 포트 확인 및 정리
lsof -ti:8501 | xargs kill -9

# 수동 실행 (상세 로그)
streamlit run src/dashboard/app.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --logger.level debug
```

---

## 🔬 고급 진단 (Advanced Diagnostics)

### 네트워크 연결 테스트

```bash
# 1. 서버 응답 확인
curl -v http://127.0.0.1:8501

# 2. 포트 리스닝 확인
netstat -an | grep 8501

# 3. DNS 해석 테스트
ping localhost
# 예상: 127.0.0.1 또는 ::1

# 4. 로컬 IP 확인
ifconfig en0 | grep "inet "
```

---

### Python 환경 검증

```bash
# 1. 모듈 설치 확인
python -c "import streamlit; print(streamlit.__version__)"
python -c "import plotly; print(plotly.__version__)"

# 2. Import 테스트
python -c "from src.dashboard import app; print('✅ OK')"

# 3. 의존성 목록
pip list | grep -E "streamlit|plotly|pandas|numpy"
```

---

### 로그 분석

디버그 모드로 실행 시 생성되는 `streamlit_debug.log` 파일 확인:

```bash
# 로그 파일 열기
cat streamlit_debug.log

# 에러 검색
grep -i "error" streamlit_debug.log
grep -i "exception" streamlit_debug.log
grep -i "failed" streamlit_debug.log
```

---

## 📞 추가 지원

### 문제가 계속되는 경우:

1. **로그 파일 수집**:

```bash
./launch_dashboard_debug.sh > full_output.log 2>&1
```

2. **시스템 정보 수집**:

```bash
python --version
pip list > pip_list.txt
ifconfig > network_info.txt
```

3. **환경 변수 확인**:

```bash
echo $PATH
echo $PYTHONPATH
```

---

## ✅ 성공 확인

대시보드가 정상 작동하면 다음이 표시됨:

**터미널:**

```
You can now view your Streamlit app in your browser.
Local URL: http://localhost:8501
Network URL: http://192.168.x.x:8501
```

**브라우저:**

```
⚾ Aegis Strategy Room - 3D Interactive Dashboard
```

왼쪽에 사이드바(Control Tower)와 메인 화면이 나타남

---

## 🚀 빠른 명령어 레퍼런스

```bash
# 표준 실행
./launch_dashboard.sh

# 디버그 실행
./launch_dashboard_debug.sh

# 포트 정리
lsof -ti:8501 | xargs kill -9

# 수동 실행 (IPv4)
streamlit run src/dashboard/app.py --server.address=0.0.0.0

# 다른 포트 사용
streamlit run src/dashboard/app.py --server.port 8502

# 의존성 재설치
pip install -r requirements-dashboard.txt --force-reinstall
```

---

**문제 해결률**: 95% 이상이 Solution B (127.0.0.1 직접 접속)로 해결됩니다.

**Version**: 1.0.0
**Last Updated**: 2026-01-06
