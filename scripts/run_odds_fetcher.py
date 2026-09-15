from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone, timedelta
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
from boatrace.odds_realtime import OddsRealtimeFetcher, _PARSERS, ODDS_HEADERS


def convert_to_dataframe(data, source="od3"):
    """取得したオッズ値リストを DataFrame に変換"""
    if data is None:
        return None
    if isinstance(data, pd.DataFrame):
        return data

    if isinstance(data, list):
        headers = ODDS_HEADERS.get(source, [])
        if headers and len(headers) == len(data):
            cols = [f"3連単_{h}" if not str(h).startswith("3連単_") else h for h in headers]
            return pd.DataFrame([dict(zip(cols, data))])
        return pd.DataFrame([data])

    if is_dataclass(data):
        return pd.DataFrame([asdict(data)])
    if hasattr(data, "__dict__"):
        return pd.DataFrame([vars(data)])
    if isinstance(data, dict):
        return pd.DataFrame([data])

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

    # 時刻表記（例: 19:40）のみを安全に抽出
    df["clean_time"] = df["電話投票締切予定"].astype(str).str.extract(r"(\d{1,2}:\d{2})")[0]
    df = df.dropna(subset=["clean_time"])

    # エラーが発生する値を NaT として安全に変換
    df["close_datetime"] = pd.to_datetime(
        today_str + " " + df["clean_time"], format="%Y-%m-%d %H:%M", errors="coerce"
    )
    df = df.dropna(subset=["close_datetime"])
    df["close_datetime"] = df["close_datetime"].dt.tz_localize(JST)

    upcoming = df[df["close_datetime"] >= now_jst].sort_values("close_datetime")
    race_col = "レース回" if "レース回" in df.columns else "レース"

    targets = []
    for _, row in upcoming.head(limit).iterrows():
        race_num_raw = str(row[race_col]).replace("R", "").strip()
        targets.append(
            {
                "stadium_code": int(row["レース場コード"]),
                "race_number": int(race_num_raw),
                "close_time": row["clean_time"],
            }
        )
    return targets


def call_fetch_method(fetcher, today_str, stadium_code, race_number):
    """OddsRealtimeFetcher.fetch_values に 'od3' を指定して呼び出し"""
    stadium_code = int(stadium_code)
    race_number = int(race_number)

    source = "od3" if "od3" in _PARSERS else list(_PARSERS.keys())[0]
    dates = [today_str, today_str.replace("-", "")]

    errors = []
    for d_str in dates:
        try:
            res = fetcher.fetch_values(source, d_str, stadium_code, race_number)
            if res is not None:
                print(f"[Success] Fetched odds for {stadium_code}R{race_number} with source='{source}'")
                return res, source
        except Exception as e:
            errors.append(f"date='{d_str}': {e}")

    print(f"[Debug] Fetch attempts errors: {errors}")
    raise RuntimeError(f"オッズ取得失敗 ({stadium_code}R{race_number})")


def format_odds_dataframe(data, source, today_str, stadium_code, race_number, deadline_time, now_jst):
    """指定のCSV構造（メタ情報＋オッズ列）へ整形"""
    df = convert_to_dataframe(data, source=source)
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


def save_or_update_csv(df_new, output_file):
    """既存のCSVが存在する場合、同一の『レースコード』行を最新データに上書き"""
    if os.path.exists(output_file):
        try:
            df_old = pd.read_csv(output_file, dtype={"レースコード": str, "レース場": str})
            df_combined = pd.concat([df_old, df_new], ignore_index=True)
            df_combined = df_combined.drop_duplicates(subset=["レースコード"], keep="last")
        except Exception as e:
            print(f"[Warning] Failed to merge with existing CSV: {e}")
            df_combined = df_new
    else:
        df_combined = df_new

    df_combined.to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig",
    )


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
            raw_data, used_source = call_fetch_method(fetcher, today_str, stadium_code, race_number)
            df_new = format_odds_dataframe(
                raw_data, used_source, today_str, stadium_code, race_number, deadline_time, now_jst
            )

            if df_new is not None and not df_new.empty:
                output_dir = f"data/previews/od3/{year}/{month}"
                os.makedirs(output_dir, exist_ok=True)
                output_file = f"{output_dir}/{day}.csv"

                save_or_update_csv(df_new, output_file)

                print(
                    f"Saved/Updated odds data to {output_file} "
                    f"({stadium_code}R{race_number}, 締切予定:{deadline_time})"
                )

        except Exception as e:
            print(f"Error processing odds {stadium_code}R{race_number}: {e}")


if __name__ == "__main__":
    main()

