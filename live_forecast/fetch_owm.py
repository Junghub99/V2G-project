"""OpenWeatherMap 5일/3시간 예보에서 구름량(연속값 %)을 받아, 기상청 예보(1시간 단위)와
같은 시간축에 맞춰 보간한다. 기상청 SKY(맑음/구름많음/흐림 3단계)보다 훨씬 세밀한 신호다."""

import pandas as pd
import requests

from config import JEJU_LAT, JEJU_LON, OWM_KEY

URL = "https://api.openweathermap.org/data/2.5/forecast"


def fetch_owm_clouds() -> pd.DataFrame:
    """datetime(3시간 간격) x clouds_pct(0~100) 표를 반환."""
    params = {"lat": JEJU_LAT, "lon": JEJU_LON, "appid": OWM_KEY, "units": "metric"}
    r = requests.get(URL, params=params, timeout=10)
    r.raise_for_status()
    items = r.json()["list"]
    rows = [{"datetime": pd.Timestamp(it["dt"], unit="s", tz="UTC").tz_convert("Asia/Seoul").tz_localize(None),
              "clouds_pct": it["clouds"]["all"]} for it in items]
    return pd.DataFrame(rows).sort_values("datetime").reset_index(drop=True)


def clouds_at_hour(owm_df: pd.DataFrame, target_dt: pd.Timestamp) -> float:
    """target_dt(1시간 단위) 시각의 구름량(%)을 3시간 간격 OWM 데이터에서 선형보간."""
    s = owm_df.set_index("datetime")["clouds_pct"]
    if target_dt <= s.index[0]:
        return float(s.iloc[0])
    if target_dt >= s.index[-1]:
        return float(s.iloc[-1])
    combined = pd.concat([s, pd.Series({target_dt: None})]).sort_index()
    combined = combined[~combined.index.duplicated(keep="first")]
    return float(combined.astype(float).interpolate(method="index")[target_dt])


if __name__ == "__main__":
    df = fetch_owm_clouds()
    print(f"OWM 예보 범위: {df['datetime'].min()} ~ {df['datetime'].max()} ({len(df)}건, 3시간 간격)\n")
    print(df.head(10).to_string())

    test_hour = df["datetime"].iloc[1] - pd.Timedelta(hours=1)
    print(f"\n보간 테스트 ({test_hour}): {clouds_at_hour(df, test_hour):.1f}%")
