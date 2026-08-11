"""기상청 단기예보 조회서비스에서 제주 지역 실시간 예보를 받아온다."""

from datetime import datetime, timedelta

import pandas as pd
import requests

from config import JEJU_NX, JEJU_NY, SERVICE_KEY

URL = "https://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst"
KMA_ISSUE_HOURS = (2, 5, 8, 11, 14, 17, 20, 23)  # 단기예보 생산시각


def _latest_issue_time(now: datetime) -> datetime:
    t = now.replace(minute=0, second=0, microsecond=0)
    while t.hour not in KMA_ISSUE_HOURS or t > now:
        t -= timedelta(hours=1)
    return t


def fetch_kma_forecast(now: datetime = None) -> pd.DataFrame:
    """제주 단기예보를 받아 datetime x [TMP, WSD, VEC, SKY, PTY] 표로 반환 (최대 약 3일치)."""
    now = now or datetime.now()
    issue_time = _latest_issue_time(now)
    params = {
        "serviceKey": SERVICE_KEY, "pageNo": 1, "numOfRows": 1000, "dataType": "JSON",
        "base_date": issue_time.strftime("%Y%m%d"), "base_time": issue_time.strftime("%H%M"),
        "nx": JEJU_NX, "ny": JEJU_NY,
    }
    r = requests.get(URL, params=params, timeout=10)
    r.raise_for_status()
    body = r.json()["response"]["body"]
    items = body["items"]["item"]

    rows = {}
    for it in items:
        if it["category"] not in ("TMP", "WSD", "VEC", "SKY", "PTY"):
            continue
        dt = pd.Timestamp(it["fcstDate"] + it["fcstTime"], tz=None)
        dt = pd.to_datetime(it["fcstDate"] + it["fcstTime"], format="%Y%m%d%H%M")
        rows.setdefault(dt, {})[it["category"]] = it["fcstValue"]

    df = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    df.index.name = "datetime"
    df = df.reset_index()
    for col in ("TMP", "WSD", "VEC", "SKY", "PTY"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["issue_time"] = issue_time
    return df


if __name__ == "__main__":
    fc = fetch_kma_forecast()
    print(f"발표시각: {fc['issue_time'].iloc[0]}  예보시간대: {fc['datetime'].min()} ~ {fc['datetime'].max()}  ({len(fc)}건)\n")
    print(fc.head(12).to_string())
