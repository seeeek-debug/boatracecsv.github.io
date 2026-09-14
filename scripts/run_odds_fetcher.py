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
from boatrace.odds_realtime import OddsRealtimeFetcher


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

    # JSTタイムゾーンを明示して締切日時を生成
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


def main():
    # 実行時の日本時間を取得
    now_jst = datetime.now(JST)
    today_str = now_jst.strftime("%Y-%m-%d")
    year, month, day = now_jst.strftime("%Y"), now_jst.strftime("%m"), now_jst.strftime("%d")

    fetcher = OddsRealtimeFetcher(rate_limiter=RateLimiter(interval_seconds=1.0))
    target_races = get_target_races(now_jst, limit=3)

    print(f"Target odds count: {len(target_races)}")

    if not target_races:
        print("対象レースが見つかりません（当日のプログラムCSVが存在しないか、全レース終了済みです）。")
        return

    for target in target_races:
        stadium_code = target["stadium_code"]
        race_number = target["race_number"]
        deadline_time = target["close_time"]

        try:
            # OddsRealtimeFetcher のメソッド呼び出し
            if hasattr(fetcher, "fetch_values"):
                data = fetcher.fetch_values(
                    date=today_str,
                    stadium_code=stadium_code,
                    race_number=race_number,
                )
            elif hasattr(fetcher, "scrape_race"):
                data = fetcher.scrape_race(
                    date=today_str,
                    stadium_code=stadium_code,
                    race_number=race_number,
                )
            else:
                print("Error: 適切なデータ取得メソッドが見つかりません。")
                break

            if data is not None:
                if isinstance(data, dict):
                    df_new = pd.DataFrame([data])
                elif isinstance(data, list):
                    df_new = pd.DataFrame(data)
                else:
                    df_new = data

                # リポジトリ構造に合わせて od3 フォルダへ保存
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

