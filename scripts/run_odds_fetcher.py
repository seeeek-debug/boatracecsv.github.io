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
    """オブジェクト/辞書/リストを DataFrame に変換"""
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


def discover_sources(fetcher):
    """OddsRealtimeFetcher 内から受け入れ可能な source を自動検出＆ソースコード出力"""
    sources = []

    # 1. クラスやモジュール内の定数属性を探索
    for attr in ["VALID_SOURCES", "SOURCES", "ALLOWED_SOURCES", "SOURCES_MAP", "ODDS_TYPES"]:
        val = getattr(fetcher, attr, None) or getattr(fetcher.__class__, attr, None)
        if val:
            if isinstance(val, (list, tuple, set)):
                sources.extend(list(val))
            elif isinstance(val, dict):
                sources.extend(list(val.keys()))

    # 2. クラスのソースコードを出力して正確な実装を確認できるようにする
    try:
        class_src = inspect.getsource(fetcher.__class__)
        print("=== [Debug] OddsRealtimeFetcher Source Code ===")
        print(class_src)
        print("===============================================")
    except Exception as e:
        print(f"[Debug] Could not get class source code: {e}")

    # 3. 候補リストのフォールバック
    fallbacks = [
        "3t", "trifecta", "3T", "3連単", "sanrentan", "3rentan",
        "3t_realtime", "realtime_3t", "3t_odds", "odds_3t",
        "official_3t", "official", "1", "3"
    ]
    for fb in fallbacks:
        if fb not in sources:
            sources.append(fb)

    return sources


def call_fetch_method(fetcher, today_str, stadium_code, race_number, valid_sources):
    """OddsRealtimeFetcher.fetch_values を検出した source で実行"""
    stadium_code = int(stadium_code)
    race_number = int(race_number)

    dates = [today_str, today_str.replace("-", "")]
    errors = []

    for src in valid_sources:
        for d_str in dates:
            try:
                res = fetcher.fetch_values(src, d_str, stadium_code, race_number)
                if res is not None:
                    print(f"[Success] Fetched with source='{src}', date='{d_str}'")
                    return res
            except Exception as e:
                errors.append(f"src='{src}', date='{d_str}': {e}")

    print(f"[Debug] Fetch attempts errors (first 5): {errors[:5]}")
    raise RuntimeError(f"オッズ取得失敗 ({stadium_code}R{race_number})")


def format_odds_dataframe(data, today_str, stadium_code, race_number, deadline_time, now_jst):
    """取得データを指定のCSV構造（メタ情報＋オッズ列）へ整形"""
    df = convert_to_dataframe(data)
    if df is None or df.empty:
        return None

    race_code = f"{today_str.replace('-', '')}{stadium_code:02d}{race_number:02d}"

    if "レースコード" not in df.columns:
        renamed = {}
        for col in df.columns:
            c_str = str(col)
            if not c_str.startswith("3連単_") and "-" in c_str:
                renamed[col] = f"3連単_{c_str}"
        if renamed:
            df = df.rename(columns=renamed)

        df.insert(0, "取得日時", now_jst.isoformat())
        df.insert(0, "締切時刻", deadline_time)
        df.insert(0, "レース回", f"{race_number:02d}R")
        df.insert(0, "レース場", f"{stadium_code:02d}")
        df.insert(0, "レース日付", today_str)
        df.insert(0, "レースコード", race_code)

    return df


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

    valid_sources = discover_sources(fetcher)
    print(f"[Debug] Candidate sources to try: {valid_sources}")

    for target in target_races:
        stadium_code = target["stadium_code"]
        race_number = target["race_number"]
        deadline_time = target["close_time"]

        try:
            raw_data = call_fetch_method(fetcher, today_str, stadium_code, race_number, valid_sources)
            df_new = format_odds_dataframe(
                raw_data, today_str, stadium_code, race_number, deadline_time, now_jst
            )

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

