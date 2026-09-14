import csv
from datetime import datetime
import os
import sys

import pandas as pd

# scriptsフォルダをモジュール検索パスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from boatrace.downloader import RateLimiter
from boatrace.race_card_scraper import RaceCardScraper  # モジュール名を修正


def main():
    now = datetime.now()
    today_str = now.strftime("%Y-%m-%d")
    year, month, day = now.strftime("%Y"), now.strftime("%m"), now.strftime("%d")

    scraper = RaceCardScraper(rate_limiter=RateLimiter(interval_seconds=1.0))
    stadium_codes = range(1, 25)  # 全24場

    print(f"Start fetching all race cards for {today_str}")

    all_rows = []
    for stadium_code in stadium_codes:
        for race_number in range(1, 13):  # 1R 〜 12R
            try:
                data = scraper.scrape_race(
                    date=today_str,
                    stadium_code=stadium_code,
                    race_number=race_number,
                )
                if data is not None:
                    if isinstance(data, pd.DataFrame):
                        all_rows.append(data)
                    elif isinstance(data, list):
                        all_rows.extend(data)
                    else:
                        all_rows.append(pd.DataFrame([data]))
            except Exception:
                # 非開催レースなどのエラーはスキップ
                pass

    # CSVファイルへの書き出し処理
    if all_rows:
        output_dir = f"data/programs/race_cards/{year}/{month}"
        os.makedirs(output_dir, exist_ok=True)
        output_file = f"{output_dir}/{day}.csv"

        combined_df = pd.concat(all_rows, ignore_index=True)
        combined_df.to_csv(output_file, index=False, encoding="utf-8-sig")

        print(
            f"Successfully saved all race cards to {output_file} ({len(combined_df)} records)"
        )
    else:
        print("No race card data found to save.")


if __name__ == "__main__":
    main()
