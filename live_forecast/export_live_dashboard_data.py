"""라이브 24시간 발전량 예측 대시보드용 JSON을 뽑는다. 백테스트가 아니라 지금 이 순간의
기상청+OpenWeatherMap 예보를 그대로 쓴다 - 실행 시점마다 값이 달라진다."""

from pathlib import Path
import json

import pandas as pd

from live_generation_forecast import live_generation_forecast

BASE_DIR = Path(__file__).resolve().parent


def build():
    df = live_generation_forecast().head(24).copy()
    generated_at = pd.Timestamp.now()

    hourly = []
    for _, r in df.iterrows():
        hourly.append({
            "datetime": r["datetime"].strftime("%Y-%m-%d %H:%M"),
            "hour_label": r["datetime"].strftime("%m/%d %H시"),
            "temp": round(float(r["기온"]), 1),
            "wind_speed": round(float(r["풍속"]), 1),
            "sky": r["하늘상태"],
            "cloud_source": r["구름소스"],
            "irradiance": round(float(r["일사량_근사"]), 2),
            "solar_mw": round(float(r["예측_태양광(MW)"]), 1),
            "wind_mw": round(float(r["예측_풍력(MW)"]), 1),
            "total_mw": round(float(r["예측_합계(MW)"]), 1),
        })

    peak = max(hourly, key=lambda h: h["total_mw"])
    trough_daytime = min(
        (h for h in hourly if 8 <= int(h["datetime"][11:13]) <= 18),
        key=lambda h: h["total_mw"], default=None,
    )

    return {
        "generated_at": generated_at.strftime("%Y-%m-%d %H:%M:%S"),
        "hourly": hourly,
        "peak": peak,
        "current_total_mw": hourly[0]["total_mw"] if hourly else None,
        "owm_hours": sum(1 for h in hourly if h["cloud_source"].startswith("OWM")),
        "sky_fallback_hours": sum(1 for h in hourly if h["cloud_source"].startswith("SKY")),
    }


if __name__ == "__main__":
    out = build()
    out_path = BASE_DIR / "live_dashboard_data.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("저장:", out_path)
    print(f"생성시각: {out['generated_at']}")
    print(f"피크: {out['peak']['hour_label']} {out['peak']['total_mw']}MW")
    print(f"OWM 사용 {out['owm_hours']}시간 / SKY 폴백 {out['sky_fallback_hours']}시간")
