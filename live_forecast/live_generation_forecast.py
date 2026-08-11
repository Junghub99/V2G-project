"""지금 시점 기준, 제주 태양광·풍력 발전량을 최대 며칠 앞까지 실시간 예측한다.

파이프라인: 기상청 실시간 단기예보(TMP/WSD/VEC/SKY/PTY) -> 일사량/일조시간 근사(irradiance_proxy)
-> wind_solar_forecast 모델의 재귀적 다단계 예측(rolling_forecast.py와 같은 방식).

한계(명시)
- "직전 몇 시간 실제 발전량" lag 입력을 받을 라이브 소스가 아직 없다. 여기서는 0으로
  placeholder 처리했다 - 그래서 예측 시작 시점(첫 1~2시간)의 정확도는 실제보다 떨어질 수
  있고, 몇 스텝 지나면 재귀적으로 스스로 안정화된다. 실시간 발전량 피드가 확보되면
  SEED_SOLAR/SEED_WIND를 그걸로 교체하면 된다.
- SMP/수요는 KPX API 승인 대기 중이라 이 스크립트에는 아직 연결하지 않았다.

구름량 소스
- OpenWeatherMap(연속값 0~100%, 3시간 간격 -> 1시간 보간)을 우선 쓰고, 실패하거나 그
  시각을 커버 못하면 기상청 SKY(3단계 범주)로 자동 대체한다.
"""

from pathlib import Path
import sys

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR.parent / "wind_solar_forecast"))
from forecast_wind_solar import predict_next_hour as gen_predict_next_hour  # noqa: E402

sys.path.insert(0, str(BASE_DIR))
from fetch_weather import fetch_kma_forecast  # noqa: E402
from fetch_owm import clouds_at_hour, fetch_owm_clouds  # noqa: E402
from irradiance_proxy import estimate_irradiance, estimate_irradiance_from_pct  # noqa: E402

GEN_MODEL_PATH = BASE_DIR.parent / "wind_solar_forecast" / "wind_solar_model.joblib"

# 라이브 실측 발전량 피드가 아직 없어 0으로 시작 (알려진 한계 - 위 docstring 참고)
SEED_SOLAR = [0.0, 0.0, 0.0]
SEED_WIND = [0.0, 0.0, 0.0]


def live_generation_forecast() -> pd.DataFrame:
    gen_model = joblib.load(GEN_MODEL_PATH)["model"]
    weather_fc = fetch_kma_forecast()

    try:
        owm = fetch_owm_clouds()
    except Exception as e:
        print(f"[경고] OpenWeatherMap 호출 실패({e}) - 기상청 SKY 범주로만 근사합니다.")
        owm = None

    solar_hist, wind_hist = list(SEED_SOLAR), list(SEED_WIND)
    rows = []
    for _, w in weather_fc.iterrows():
        dt = w["datetime"]
        cloud_source = "SKY(범주)"
        if owm is not None and owm["datetime"].min() <= dt <= owm["datetime"].max():
            cloud_pct = clouds_at_hour(owm, dt)
            irr = estimate_irradiance_from_pct(cloud_pct, int(w["PTY"]), dt)
            cloud_source = f"OWM({cloud_pct:.0f}%)"
        else:
            irr = estimate_irradiance(int(w["SKY"]), int(w["PTY"]), dt)

        forecast_weather = {
            "datetime": dt,
            "기온(C)": w["TMP"], "풍속(m/s)": w["WSD"], "풍향(deg)": w["VEC"],
            "일사량(MJ/m2)": irr["일사량(MJ/m2)"], "일조시간(hr)": irr["일조시간(hr)"],
            "전운량(0-10)": irr["전운량(0-10)_근사"],
        }
        pred = gen_predict_next_hour(gen_model, forecast_weather, solar_hist[:3], wind_hist[:3])
        solar_hist.insert(0, pred["solar_gen"])
        wind_hist.insert(0, pred["wind_gen"])

        rows.append({
            "datetime": dt, "기온": w["TMP"], "풍속": w["WSD"],
            "하늘상태": {1: "맑음", 3: "구름많음", 4: "흐림"}.get(int(w["SKY"]), w["SKY"]),
            "구름소스": cloud_source,
            "일사량_근사": round(irr["일사량(MJ/m2)"], 2),
            "예측_태양광(MW)": round(pred["solar_gen"], 1),
            "예측_풍력(MW)": round(pred["wind_gen"], 1),
            "예측_합계(MW)": round(pred["total_gen"], 1),
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    result = live_generation_forecast()
    print(f"기준시각(지금): {pd.Timestamp.now()}\n")
    print(result.head(24).to_string(index=False))
    out_path = BASE_DIR / "live_generation_forecast.csv"
    result.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n저장: {out_path}")
