"""
2022년 제주 기상데이터 + 풍력/태양광 발전량 데이터를 이용해,
"1시간 뒤 기상 예보 + 직전 몇 시간 실제 발전량이 주어졌을 때 1시간 뒤 풍력/태양광/합산 발전량"을
예측하는 회귀 모델. 여러 모델(RandomForest / GradientBoosting / HistGradientBoosting)을 비교해
가장 성능이 좋은 모델을 최종 선택한다.

입력 데이터 (V2G 폴더 루트)
- jeju_기상데이터_hourly_2022.csv : 제주/고산/성산/서귀포 4개 관측소, 시간별 기상데이터
- 2022 제주 태양광.csv            : 일자 x 1~24시 형태의 태양광 발전량(MW)
- 2022 제주 풍력.csv              : 일자 x 1~24시 형태의 풍력 발전량(MW)

핵심 아이디어
- 발전량은 그 시각의 기상조건에 좌우되지만, 풍력은 관성/기압계 이동 등으로 시간적 상관도 크다.
  그래서 "그 시각 기상" 특징에 더해 "직전 1~3시간 실제 발전량(lag)"과 "직전 3시간 평균"을
  추가 입력으로 사용해 정확도를 끌어올린다.
- 예측 시점에는 (1시간 뒤 예보 기상 + 현재까지 알고 있는 실제 발전량 이력)을 모델에 넣으면
  1시간 뒤 발전량 예측이 된다.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor

DATA_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(__file__).resolve().parent / "wind_solar_model.joblib"

WEATHER_COLS = ["기온(C)", "풍속(m/s)", "풍향(deg)", "일사량(MJ/m2)", "일조시간(hr)", "전운량(0-10)"]

WEATHER_FEATURE_COLS = [
    "기온(C)", "풍속(m/s)", "일사량(MJ/m2)", "일조시간(hr)", "전운량(0-10)",
    "wind_dir_sin", "wind_dir_cos",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
]
LAG_STEPS = [1, 2, 3]
LAG_TARGETS = ["solar_gen", "wind_gen"]
LAG_FEATURE_COLS = [f"{t}_lag{k}" for t in LAG_TARGETS for k in LAG_STEPS] + \
                    [f"{t}_roll3" for t in LAG_TARGETS]
FEATURE_COLS = WEATHER_FEATURE_COLS + LAG_FEATURE_COLS
TARGET_COLS = ["solar_gen", "wind_gen"]


def load_weather(year: int = 2022) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "jeju_기상데이터_hourly_2022.csv", encoding="utf-8-sig")
    df["관측시각"] = pd.to_datetime(df["관측시각"])
    df = df[df["관측시각"].dt.year == year].copy()

    # 야간에는 일사량/일조시간이 결측으로 기록됨 -> 실제 값은 0
    df["일사량(MJ/m2)"] = df["일사량(MJ/m2)"].fillna(0)
    df["일조시간(hr)"] = df["일조시간(hr)"].fillna(0)

    # 나머지 결측(관측 누락)은 관측소별 시계열 선형보간으로 채움
    for col in ["기온(C)", "풍속(m/s)", "풍향(deg)", "전운량(0-10)"]:
        df[col] = df.groupby("지점명")[col].transform(lambda s: s.interpolate().ffill().bfill())

    # 풍향은 각도이므로 산술평균이 불가능 -> sin/cos으로 변환 후 평균
    rad = np.deg2rad(df["풍향(deg)"])
    df["wind_dir_sin"] = np.sin(rad)
    df["wind_dir_cos"] = np.cos(rad)

    # 4개 관측소를 제주 지역 평균 기상값 하나로 집계 (풍력/태양광 발전량은 도 단위 합산치이므로)
    agg = df.groupby("관측시각")[["기온(C)", "풍속(m/s)", "일사량(MJ/m2)", "일조시간(hr)",
                                "전운량(0-10)", "wind_dir_sin", "wind_dir_cos"]].mean()
    agg = agg.reset_index().rename(columns={"관측시각": "datetime"})
    # 참고/표시용으로 각도값도 복원 (모델 입력 자체는 sin/cos 사용)
    agg["풍향(deg)"] = np.rad2deg(np.arctan2(agg["wind_dir_sin"], agg["wind_dir_cos"])) % 360
    return agg


def _wide_to_long(path: Path, value_name: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df.rename(columns={df.columns[0]: "날짜"})
    hour_cols = [c for c in df.columns if c.endswith("시")]
    long_df = df.melt(id_vars=["날짜"], value_vars=hour_cols, var_name="hour_label", value_name=value_name)
    # "1시" ~ "24시" 는 00:00 ~ 23:00 시각의 발전량 (24시 -> 23시 슬롯)
    long_df["hour"] = long_df["hour_label"].str.replace("시", "", regex=False).astype(int) - 1
    long_df["datetime"] = pd.to_datetime(long_df["날짜"]) + pd.to_timedelta(long_df["hour"], unit="h")
    return long_df[["datetime", value_name]].sort_values("datetime").reset_index(drop=True)


def load_generation() -> pd.DataFrame:
    solar = _wide_to_long(DATA_DIR / "2022 제주 태양광.csv", "solar_gen")
    wind = _wide_to_long(DATA_DIR / "2022 제주 풍력.csv", "wind_gen")
    gen = solar.merge(wind, on="datetime")
    gen["total_gen"] = gen["solar_gen"] + gen["wind_gen"]
    return gen


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    hour = df["datetime"].dt.hour
    month = df["datetime"].dt.month
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("datetime").reset_index(drop=True)
    for target in LAG_TARGETS:
        for k in LAG_STEPS:
            df[f"{target}_lag{k}"] = df[target].shift(k)
        # roll3은 직전 3시간 평균 (현재 시각 값은 포함하지 않음 -> shift(1) 기준)
        df[f"{target}_roll3"] = df[target].shift(1).rolling(3).mean()
    return df.dropna(subset=LAG_FEATURE_COLS).reset_index(drop=True)


def build_dataset() -> pd.DataFrame:
    weather = load_weather()
    gen = load_generation()
    data = weather.merge(gen, on="datetime", how="inner")
    data = add_time_features(data)
    data = add_lag_features(data)
    return data.sort_values("datetime").reset_index(drop=True)


def _build_candidate_models() -> dict:
    return {
        "RandomForest": RandomForestRegressor(
            n_estimators=400, min_samples_leaf=2, random_state=42, n_jobs=-1,
        ),
        "GradientBoosting": MultiOutputRegressor(
            GradientBoostingRegressor(
                n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42,
            )
        ),
        "HistGradientBoosting": MultiOutputRegressor(
            HistGradientBoostingRegressor(
                max_iter=300, learning_rate=0.05, max_depth=6, random_state=42,
            )
        ),
    }


def _evaluate(test: pd.DataFrame, pred: pd.DataFrame) -> dict:
    metrics = {}
    for col in TARGET_COLS:
        metrics[col] = {
            "RMSE": mean_squared_error(test[col], pred[col]) ** 0.5,
            "MAE": mean_absolute_error(test[col], pred[col]),
            "R2": r2_score(test[col], pred[col]),
        }
    total_pred = pred["solar_gen"] + pred["wind_gen"]
    metrics["total_gen"] = {
        "RMSE": mean_squared_error(test["total_gen"], total_pred) ** 0.5,
        "MAE": mean_absolute_error(test["total_gen"], total_pred),
        "R2": r2_score(test["total_gen"], total_pred),
    }
    return metrics


def train_and_compare(data: pd.DataFrame, test_months=(11, 12)):
    is_test = data["datetime"].dt.month.isin(test_months)
    train, test = data[~is_test], data[is_test]
    print(f"[학습 {len(train)}건 / 검증 {len(test)}건 ({test_months}월 홀드아웃)]\n")

    fitted, all_metrics = {}, {}
    for name, model in _build_candidate_models().items():
        model.fit(train[FEATURE_COLS], train[TARGET_COLS])
        pred = pd.DataFrame(
            model.predict(test[FEATURE_COLS]), columns=TARGET_COLS, index=test.index
        ).clip(lower=0)
        metrics = _evaluate(test, pred)
        fitted[name] = model
        all_metrics[name] = metrics

        print(f"[{name}]")
        for col in TARGET_COLS + ["total_gen"]:
            m = metrics[col]
            print(f"  {col:10s}  RMSE={m['RMSE']:7.2f}  MAE={m['MAE']:7.2f}  R2={m['R2']:.3f}")
        print()

    best_name = min(all_metrics, key=lambda n: all_metrics[n]["total_gen"]["RMSE"])
    print(f"-> total_gen RMSE 기준 최적 모델: {best_name}")

    best_model = fitted[best_name]
    if hasattr(best_model, "feature_importances_"):
        importance = pd.Series(best_model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
        print(f"\n[{best_name} 변수 중요도]")
        print(importance.to_string())

    return best_name, best_model, fitted, all_metrics


def _weather_features(weather_row: dict) -> dict:
    dt = pd.Timestamp(weather_row["datetime"])
    rad = np.deg2rad(weather_row["풍향(deg)"])
    return {
        "기온(C)": weather_row["기온(C)"],
        "풍속(m/s)": weather_row["풍속(m/s)"],
        "일사량(MJ/m2)": weather_row["일사량(MJ/m2)"],
        "일조시간(hr)": weather_row["일조시간(hr)"],
        "전운량(0-10)": weather_row["전운량(0-10)"],
        "wind_dir_sin": np.sin(rad),
        "wind_dir_cos": np.cos(rad),
        "hour_sin": np.sin(2 * np.pi * dt.hour / 24),
        "hour_cos": np.cos(2 * np.pi * dt.hour / 24),
        "month_sin": np.sin(2 * np.pi * dt.month / 12),
        "month_cos": np.cos(2 * np.pi * dt.month / 12),
    }


def predict_next_hour(model, forecast_weather: dict, recent_solar_gen: list, recent_wind_gen: list) -> dict:
    """
    forecast_weather : 예측하려는 1시간 뒤 시각의 예보 기상 dict (WEATHER_COLS + datetime)
    recent_solar_gen / recent_wind_gen : [현재(t) 실제값, t-1 실제값, t-2 실제값] (최신순 3개)
                                          -> lag1/lag2/lag3/roll3 계산에 사용
    """
    assert len(recent_solar_gen) == 3 and len(recent_wind_gen) == 3, "최근 3시간 실제 발전량이 필요합니다."

    row = {**_weather_features(forecast_weather)}
    for k in LAG_STEPS:
        row[f"solar_gen_lag{k}"] = recent_solar_gen[k - 1]
        row[f"wind_gen_lag{k}"] = recent_wind_gen[k - 1]
    row["solar_gen_roll3"] = float(np.mean(recent_solar_gen))
    row["wind_gen_roll3"] = float(np.mean(recent_wind_gen))

    x = pd.DataFrame([row])[FEATURE_COLS]
    pred = model.predict(x)[0]
    solar, wind = max(pred[0], 0.0), max(pred[1], 0.0)
    return {"solar_gen": solar, "wind_gen": wind, "total_gen": solar + wind}


if __name__ == "__main__":
    dataset = build_dataset()
    print(f"데이터셋 크기: {dataset.shape}, 기간: {dataset['datetime'].min()} ~ {dataset['datetime'].max()}\n")

    best_name, best_model, fitted, all_metrics = train_and_compare(dataset)
    joblib.dump({"model": best_model, "name": best_name}, MODEL_PATH)
    print(f"\n모델 저장 완료 ({best_name}): {MODEL_PATH}")

    # 사용 예시: (현재 t 시각까지의 실제 발전량 이력) + (t+1 예보 기상) -> t+1 발전량 예측
    row_next = dataset.iloc[-1]                      # t+1 (예측 대상 시각)
    hist = dataset.iloc[-4:-1].sort_values("datetime", ascending=False)  # t, t-1, t-2

    forecast_weather = {"datetime": row_next["datetime"], **{c: row_next[c] for c in WEATHER_COLS}}
    recent_solar = hist["solar_gen"].tolist()
    recent_wind = hist["wind_gen"].tolist()

    pred = predict_next_hour(best_model, forecast_weather, recent_solar, recent_wind)
    print("\n[예측 예시]")
    print(f"기준 시각(t): {hist.iloc[0]['datetime']}  실제 solar={recent_solar[0]:.2f}, wind={recent_wind[0]:.2f}")
    print(f"예측 대상(t+1): {row_next['datetime']}")
    print(f"  예측값: {pred}")
    print(f"  실제값: solar={row_next['solar_gen']:.2f}, wind={row_next['wind_gen']:.2f}, "
          f"total={row_next['total_gen']:.2f}")
