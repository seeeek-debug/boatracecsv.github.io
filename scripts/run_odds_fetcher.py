from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone, timedelta
import inspect
import os
from pathlib import Path
import sys

import pandas as pd

# 日本時間（JST: UTC+9）の定義
JST = timezone(timedelta(hours=9))

# プロジェクトルートをパスの先頭に追加
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from boatrace.downloader import RateLimiter
from boatrace.odds_realtime import OddsRealtimeFetcher


def convert_to_dataframe(data):
    """オブジェクト/辞書/リストを DataFrame に安全変換"""
    if data is None:
        return None
    if isinstance(data, pd.DataFrame):
        return data
    if is_dataclass(data):
        return pd.DataFrame([asdict(data)])
    if hasattr(data, "__dict__"):
        return pd.DataFrame([vars(data)])
    if isinstance(data, dict):
        return pd.DataFrame([data])
    if isinstance(data, list):
        rows = [
            asdict(x) if is_dataclass(x) else (vars(x) if hasattr(x, "__dict__") else x)
            for x in data
        ]
        return pd.DataFrame(rows)
    return pd.DataFrame([data])


def get_target_races(now_jst, limit=3):
    """日本時間（JST）ベースで当日のプログラムCSVから直近Nレースを取得"""
    today_str = now_jst.strftime("%Y-%m-%d")
    year, month, day = now_jst.strftime("%Y"), now_jst.strftime("%m"), now_jst.strftime("%d")

    csv_path = f"data/programs/title/{year}/{month}/{day}.csv"

    if not os.path.exists(csv_path):
        print(f"Program CSV not found: {csv_path}")
        return []

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return []

    df.columns = df.columns.str.strip()

    if "電話投票締切予定" not in df.columns:
        print(f"利用可能な列名一覧: {list(df.columns)}")
        return []

    df["close_datetime"] = pd.to_datetime(
        today_str + " " + df["電話投票締切予定"], format="%Y-%m-%d %H:%M"
    ).dt.tz_localize(JST)

    upcoming = df[df["close_datetime"] >= now_jst].sort_values("close_datetime")
    race_col = "レース回" if "レース回" in df.columns else "レース"

    targets = []
    for _, row in upcoming.head(limit).iterrows():
        race_num_raw = str(row[race_col]).replace("R", "").strip()
        targets.append(
            {
                "stadium_code": int(row["レース場コード"]),
                "race_number": int(race_num_raw),
                "close_time": row["電話投票締切予定"],
            }
        )
    return targets


def call_fetch_method(fetcher, today_str, stadium_code, race_number):
    """OddsRealtimeFetcher の引数定義（source, date_str 等）に合わせて動的に呼び出し"""
    methods_to_try = ["fetch_values", "scrape_race", "fetch", "scrape"]
    
    for method_name in methods_to_try:
        if not hasattr(fetcher, method_name):
            continue
        
        method = getattr(fetcher, method_name)
        sig = inspect.signature(method)
        params = sig.parameters

        # キーワード引数の組み立て
        kwargs = {}
        for param_name in params:
            if param_name in ["self", "cls"]:
                continue
            if param_name in ["source", "src", "odds_type", "type"]:
                kwargs[param_name] = "official"
            elif param_name in ["date_str", "date", "race_date", "ymd"]:
                kwargs[param_name] = today_str
            elif param_name in ["stadium_code", "stadium", "jyo_code", "place_code"]:
                kwargs[param_name] = stadium_code
            elif param_name in ["race_number", "race", "race_num", "race_no"]:
                kwargs[param_name] = race_number

        try:
            res = method(**kwargs)
            if res is not None:
                return res
        except Exception:
            pass

        # 位置引数での呼び出しを試行（フォールバック）
        positional_patterns = [
            ("official", today_str, stadium_code, race_number),
            (today_str, stadium_code, race_number),
            (stadium_code, race_number),
        ]
        for pattern in positional_patterns:
            try:
                res = method(*pattern)
                if res is not None:
                    return res
            except Exception:
                pass

    raise RuntimeError("OddsRealtimeFetcher の取得メソッド呼び出しに失敗しました。")


def main():
    now_jst = datetime.now(JST)
    today_str = now_jst.strftime("%Y-%m-%d")
    year, month, day = now_jst.strftime("%Y"), now_jst.strftime("%m"), now_jst.strftime("%d")

    fetcher = OddsRealtimeFetcher(rate_limiter=RateLimiter(interval_seconds=1.0))
    target_races = get_target_races(now_jst, limit=3)

    print(f"Target odds count: {len(target_races)}")

    if not target_races:
        print("対象レースが見つかりません。")
        return

    for target in target_races:
        stadium_code = target["stadium_code"]
        race_number = target["race_number"]
        deadline_time = target["close_time"]

        try:
            data = call_fetch_method(fetcher, today_str, stadium_code, race_number)
            df_new = convert_to_dataframe(data)

            if df_new is not None and not df_new.empty:
                output_dir = f"data/previews/od3/{year}/{month}"
                os.makedirs(output_dir, exist_ok=True)
                output_file = f"{output_dir}/{day}.csv"

                file_exists = os.path.exists(output_file)
                df_new.to_csv(
                    output_file,
                    mode="a" if file_exists else "w",
                    header=not file_exists,
                    index=False,
                    encoding="utf-8-sig",
                )

                print(
                    f"Saved odds data to {output_file} "
                    f"({stadium_code}R{race_number}, 締切予定:{deadline_time})"
                )

        except Exception as e:
            print(f"Error processing odds {stadium_code}R{race_number}: {e}")


if __name__ == "__main__":
    main()

