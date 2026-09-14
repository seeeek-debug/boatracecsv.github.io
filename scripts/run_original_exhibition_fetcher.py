import csv
from datetime import datetime
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from boatrace.downloader import RateLimiter
from boatrace.original_exhibition_scraper import OriginalExhibitionScraper


def get_target_races(limit=3):
    """当日のプログラムCSVから、現在時刻以降で最も締め切りが近い直近Nレースを取得"""
    now = datetime.now()
    today_str = now.strftime("%Y-%m-%d")
    year, month, day = now.strftime("%Y"), now.strftime("%m"), now.strftime("%d")

    csv_path = f"data/programs/title/{year}/{month}/{day}.csv"
    if not os.path.exists(csv_path):
        return []

    import pandas as pd

    df = pd.read_csv(csv_path)
    df["close_datetime"] = pd.to_datetime(
        today_str + " " + df["電話投票締切予定"], format="%Y-%m-%d %H:%M"
    )
    upcoming = df[df["close_datetime"] >= now].sort_values("close_datetime")

    targets = []
    for _, row in upcoming.head(limit).iterrows():
        targets.append(
            {
                "stadium_code": int(row["レース場コード"]),
                "race_number": int(row["レース"]),
                "close_time": row["電話投票締切予定"],
            }
        )
    return targets


def main():
    now = datetime.now()
    today_str = now.strftime("%Y-%m-%d")
    year, month, day = now.strftime("%Y"), now.strftime("%m"), now.strftime("%d")

    scraper = OriginalExhibitionScraper(
        rate_limiter=RateLimiter(interval_seconds=1.0)
    )
    target_races = get_target_races(limit=3)

    print(f"Target original exhibition count: {len(target_races)}")

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
                # 画像の構造に合わせた保存先パス（日別ファイルに追記・保存）
                output_dir = f"data/previews/original_exhibition/{year}/{month}"
                os.makedirs(output_dir, exist_ok=True)
                output_file = f"{output_dir}/{day}.csv"

                print(
                    f"Saved exhibition data to {output_file} ({stadium_code}R{race_number})"
                )

        except Exception as e:
            print(
                f"Error processing exhibition {stadium_code}R{race_number}: {e}"
            )


if __name__ == "__main__":
    main()
