"""라이브 발전량 데이터를 새로 뽑아서 live_dashboard.html의 DATA를 그 자리에서 갱신한다.
스케줄러가 이 스크립트 하나만 실행하면 되도록 fetch+inject를 한 번에 처리한다.

사용법: python refresh_live_dashboard.py
(이 파일이 있는 디렉터리에서 실행 - live_generation_forecast, fetch_owm 등을 상대 import한다)
"""

from pathlib import Path
import json
import re

from export_live_dashboard_data import build

BASE_DIR = Path(__file__).resolve().parent
HTML_PATH = BASE_DIR / "live_dashboard.html"
DATA_JSON_PATH = BASE_DIR / "live_dashboard_data.json"


def refresh():
    data = build()

    with open(DATA_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    html = HTML_PATH.read_text(encoding="utf-8")
    new_line = "  const DATA = " + json.dumps(data, ensure_ascii=False) + ";"
    html, n = re.subn(r"^  const DATA = .*;$", new_line, html, count=1, flags=re.MULTILINE)
    if n != 1:
        raise RuntimeError(f"'const DATA = ...;' 줄을 정확히 1개 찾지 못했습니다 (찾은 개수: {n}) - "
                            f"live_dashboard.html 구조가 바뀌었는지 확인하세요.")
    HTML_PATH.write_text(html, encoding="utf-8")

    return data


if __name__ == "__main__":
    data = refresh()
    print(f"갱신 완료: {HTML_PATH}")
    print(f"생성시각: {data['generated_at']}")
    print(f"피크: {data['peak']['hour_label']} {data['peak']['total_mw']}MW")
    print(f"OWM 사용 {data['owm_hours']}시간 / SKY 폴백 {data['sky_fallback_hours']}시간")
