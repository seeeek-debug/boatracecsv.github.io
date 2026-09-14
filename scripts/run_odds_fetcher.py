from datetime import datetime
import os

from pathlib import Path
import sys

import pandas as pd

# プロジェクトルートディレクトリをパスの先頭に追加
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from boatrace.downloader import RateLimiter
# odds.py ではなく odds_realtime.py からインポート
from boatrace.odds_realtime import (
    RealtimeOddsScraper,  # クラス名が異なる場合はモジュール内の対応するクラス名に調整してください
)


def get_target_races(limit=3):
    """当日のプログラムCSVから、現在時刻以降で最も締め切りが近い直近Nレースを取得"""
    now = datetime.now()
    today_str = now.strftime("%Y-%m-%d")
    year, month, day = now.strftime("%Y"), now.strftime("%m"), now.strftime("%d")

    csv_path = f"data/programs/title/{year}/{month}/{day}.csv"

    if not os.path.exists(csv_path):
        print(f"Program CSV not found: {csv_path}")
        return []

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return []

    # 列名の前後の空白を削除
    df.columns = df.columns.str.strip()

    if "電話投票締切予定" not in df.columns:
        print(f"利用可能な列名一覧: {list(df.columns)}")
        return []

    df["close_datetime"] = pd.to_datetime(
        today_str + " " + df["電話投票締切予定"], format="%Y-%m-%d %H:%M"
    )

    upcoming = df[df["close_datetime"] >= now].sort_values("close_datetime")

    # 列名の対応: "レース回" (例: "1R") または "レース" の両方に対応
    race_col = "レース回" if "レース回" in df.columns else "レース"

    targets = []
    for _, row in upcoming.head(limit).iterrows():
        # "1R" などの文字列から数字のみを抽出
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
    now = datetime.now()
    today_str = now.strftime("%Y-%m-%d")
    year, month, day = now.strftime("%Y"), now.strftime("%m"), now.strftime("%d")

    # RealtimeOddsScraper のインスタンス化
    scraper = RealtimeOddsScraper(rate_limiter=RateLimiter(interval_seconds=1.0))
    target_races = get_target_races(limit=3)

    print(f"Target odds count: {len(target_races)}")

    for target in target_races:
        stadium_code = target["stadium_code"]
        race_number = target["race_number"]
        deadline_time = target["close_time"]

        try:
            data = scraper.scrape_race(
                date=today_str,
                stadium_code=stadium_code,
                race_number=race_number,
            )

            if data is not None:
                if isinstance(data, dict):
                    df_new = pd.DataFrame([data])
                elif isinstance(data, list):
                    df_new = pd.DataFrame(data)
                else:
                    df_new = data

                output_dir = f"data/previews/odds/{year}/{month}"
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

