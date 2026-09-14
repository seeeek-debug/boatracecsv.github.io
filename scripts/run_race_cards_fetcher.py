import csv
from datetime import datetime
import os
import sys

import pandas as pd

# scriptsフォルダをモジュール検索パスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from boatrace.downloader import RateLimiter
from boatrace.race_card import RaceCardScraper


def get_target_races(limit=5):
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

    df["close_datetime"] = pd.to_datetime(
        today_str + " " + df["電話投票締切"], format="%Y-%m-%d %H:%M"
    )

    upcoming = df[df["close_datetime"] >= now].sort_values("close_datetime")

    targets = []
    for _, row in upcoming.head(limit).iterrows():
        targets.append(
            {
                "stadium_code": int(row["レース場コード"]),
                "race_number": int(row["レース"]),
                "close_time": row["電話投票締切"],
            }
        )
    return targets


def main():
    today_str = datetime.now().strftime("%Y-%m-%d")
    year, month, _ = today_str.split("-")

    scraper = RaceCardScraper(rate_limiter=RateLimiter(interval_seconds=1.0))

    # 直近5レースを取得対象に指定
    target_races = get_target_races(limit=5)
    print(f"Target race cards count: {len(target_races)}")

    for target in target_races:
        stadium_code = target["stadium_code"]
        race_number = target["race_number"]

        try:
            data = scraper.scrape_race(
                date=today_str,
                stadium_code=stadium_code,
                race_number=race_number,
            )

            if data:
                output_dir = f"data/race-cards/{year}/{month}"
                os.makedirs(output_dir, exist_ok=True)
                output_file = f"{output_dir}/{today_str.replace('-', '')}_{stadium_code:02d}_{race_number:02d}.csv"

                # データをCSVへ書き出す処理
                # ※実装に合わせてデータフレーム出力やファイル保存関数を適用してください
                print(
                    f"Saved race card: {stadium_code}R{race_number} -> {output_file}"
                )

        except Exception as e:
            print(
                f"Error processing race card {stadium_code}R{race_number}: {e}"
            )


if __name__ == "__main__":
    main()

