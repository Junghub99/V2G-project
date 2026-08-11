"""기상청 단기예보의 범주형 하늘상태(SKY: 맑음/구름많음/흐림)를, 우리 발전량 모델이 필요로 하는
연속값 일사량(MJ/m2)·일조시간(hr)으로 근사 변환한다.

방법
- 기상청 단기예보엔 일사량이 없다. 대신 2022년 실측 데이터(jeju_기상데이터_hourly_2022.csv)에는
  전운량(0~10)과 실측 일사량·일조시간이 같이 있으므로, "전운량 + 시간대 + 계절 -> 일사량/일조시간"
  관계를 이 실측 데이터로 직접 학습해둔다.
- 그 다음 SKY 범주(맑음/구름많음/흐림)를 전운량 대표값으로 매핑해서 위 모델에 넣으면, 근사지만
  나름 데이터 기반의 일사량 추정치가 나온다. (기상청 SKY 구간 경계가 공식적으로 정확히
  공개돼 있진 않아, 일반적으로 쓰이는 근사 구간의 중간값을 썼다 - 맑음≈2.5할, 구름많음≈6.5할,
  흐림≈9.5할.)
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR.parent / "wind_solar_forecast"))
from forecast_wind_solar import load_weather  # noqa: E402

SKY_TO_CLOUD_AMOUNT = {1: 2.5, 3: 6.5, 4: 9.5}  # 기상청 SKY 코드 -> 전운량(0~10) 대표값 근사
PTY_RAIN_CODES = {1, 2, 3, 4}  # 강수형태: 0=없음 이외는 비/눈 -> 일사량 추가 감쇠

_model_cache = {}


def _train_proxy_model():
    """전운량+시간대+계절 -> 일사량/일조시간 회귀모델을 실측 데이터로 학습(1회, 캐시)."""
    if "model" in _model_cache:
        return _model_cache["model"]

    weather = load_weather()
    hour = weather["datetime"].dt.hour
    month = weather["datetime"].dt.month
    X = pd.DataFrame({
        "cloud": weather["전운량(0-10)"],
        "hour_sin": np.sin(2 * np.pi * hour / 24), "hour_cos": np.cos(2 * np.pi * hour / 24),
        "month_sin": np.sin(2 * np.pi * month / 12), "month_cos": np.cos(2 * np.pi * month / 12),
    })
    y = weather[["일사량(MJ/m2)", "일조시간(hr)"]]

    model = RandomForestRegressor(n_estimators=200, min_samples_leaf=5, random_state=42, n_jobs=-1)
    model.fit(X, y)
    _model_cache["model"] = model
    return model


def _predict_from_cloud(cloud_0_10: float, dt: pd.Timestamp) -> dict:
    model = _train_proxy_model()
    x = pd.DataFrame([{
        "cloud": cloud_0_10,
        "hour_sin": np.sin(2 * np.pi * dt.hour / 24), "hour_cos": np.cos(2 * np.pi * dt.hour / 24),
        "month_sin": np.sin(2 * np.pi * dt.month / 12), "month_cos": np.cos(2 * np.pi * dt.month / 12),
    }])
    pred = model.predict(x)[0]
    return {
        "일사량(MJ/m2)": max(float(pred[0]), 0.0),
        "일조시간(hr)": max(float(pred[1]), 0.0),
        "전운량(0-10)_근사": cloud_0_10,
    }


def estimate_irradiance(sky_code: int, pty_code: int, dt: pd.Timestamp) -> dict:
    """기상청 SKY/PTY 코드(3단계 범주) + 시각 -> 일사량/일조시간 근사치."""
    cloud = SKY_TO_CLOUD_AMOUNT.get(int(sky_code), 6.5)
    if int(pty_code) in PTY_RAIN_CODES:
        cloud = min(cloud + 1.5, 10.0)  # 비/눈 오면 구름이 더 낀 것으로 취급
    return _predict_from_cloud(cloud, dt)


def estimate_irradiance_from_pct(cloud_pct_0_100: float, pty_code: int, dt: pd.Timestamp) -> dict:
    """OpenWeatherMap 구름량(연속값 0~100%) + 시각 -> 일사량/일조시간 근사치.
    SKY 3단계보다 세밀한 신호라 estimate_irradiance()보다 정확도가 높다."""
    cloud = cloud_pct_0_100 / 10.0  # 0~100% -> 0~10할
    if int(pty_code) in PTY_RAIN_CODES:
        cloud = min(cloud + 1.5, 10.0)
    return _predict_from_cloud(cloud, dt)


if __name__ == "__main__":
    # 검증: 맑은 대낮 vs 흐린 대낮 vs 밤 이 상식적인 방향으로 나오는지 확인
    noon = pd.Timestamp("2026-08-11 13:00")
    night = pd.Timestamp("2026-08-11 22:00")
    print("맑음(SKY=1) 낮 13시 :", estimate_irradiance(1, 0, noon))
    print("구름많음(SKY=3) 낮 13시:", estimate_irradiance(3, 0, noon))
    print("흐림(SKY=4) 낮 13시  :", estimate_irradiance(4, 0, noon))
    print("비(PTY=1) 낮 13시   :", estimate_irradiance(4, 1, noon))
    print("맑음 밤 22시        :", estimate_irradiance(1, 0, night))
