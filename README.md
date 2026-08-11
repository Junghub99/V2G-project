# 제주 신재생 발전량 라이브 예측

기상청 단기예보 + OpenWeatherMap 구름량을 받아 제주 태양광·풍력 24시간 발전량을 재귀예측하고,
`live_forecast/live_dashboard.html`을 갱신한다.

## 실행 방법

```bash
pip install -r requirements.txt
cp live_forecast/config.py.example live_forecast/config.py
# config.py를 열어 SERVICE_KEY(기상청/data.go.kr), OWM_KEY(OpenWeatherMap) 실제 값 채우기
python live_forecast/refresh_live_dashboard.py
```

성공하면 `live_forecast/live_dashboard.html`이 최신 데이터로 갱신된다. 이 파일을 Claude 아티팩트로
게시(또는 재게시)하면 대시보드가 갱신된다.

## 매시간 자동 갱신 (GitHub Actions + Pages)

`.github/workflows/refresh.yml`이 매시간 자동으로 위 스크립트를 실행하고 결과를 커밋한다.
처음 한 번만 아래 설정이 필요하다.

1. **저장소 Secrets 등록** — Settings → Secrets and variables → Actions → New repository secret
   - `KMA_SERVICE_KEY` : 기상청 단기예보 서비스키 (data.go.kr)
   - `OWM_API_KEY` : OpenWeatherMap API 키

2. **GitHub Pages 활성화** — Settings → Pages → Source: "Deploy from a branch" → Branch: `master` / `(root)` → Save

3. 설정 후 첫 실행은 Actions 탭 → "Refresh live dashboard" → "Run workflow"로 수동 실행해 확인 가능.
   이후엔 매시간 자동 실행된다.

갱신되는 페이지 주소: `https://<github-id>.github.io/<repo-이름>/live_forecast/live_dashboard.html`

## 구성

- `live_forecast/fetch_weather.py` — 기상청 단기예보 조회 (제주시 nx=53, ny=38)
- `live_forecast/fetch_owm.py` — OpenWeatherMap 5일/3시간 예보에서 구름량(연속값 %) 조회
- `live_forecast/irradiance_proxy.py` — 구름량 → 일사량/일조시간 근사 (2022년 실측 데이터로 학습)
- `live_forecast/live_generation_forecast.py` — 발전량 재귀예측 파이프라인
- `live_forecast/export_live_dashboard_data.py` — 대시보드용 JSON 생성
- `live_forecast/refresh_live_dashboard.py` — 위 전체를 실행하고 `live_dashboard.html`에 주입
- `wind_solar_forecast/` — 학습된 발전량 예측 모델(RandomForest/GradientBoosting 등)
- `jeju_기상데이터_hourly_2022.csv` — 일사량 근사 모델 학습용 실측 기상데이터
