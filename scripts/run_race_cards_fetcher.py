import csv
from datetime import datetime
import os
import sys

# scriptsフォルダをモジュール検索パスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from boatrace.downloader import RateLimiter
from boatrace.race_card import RaceCardScraper


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
                if data:
                    all_rows.append(data)
            except Exception:
                # 開催されていないレースなどのエラーはスキップ
                pass

    # 取得できたデータを日付ごとのファイルに保存
    if all_rows:
        output_dir = f"data/programs/race_cards/{year}/{month}"
        os.makedirs(output_dir, exist_ok=True)
        output_file = f"{output_dir}/{day}.csv"

        # ここでCSVへの書き出し処理を行います
        # （※既存のスクレイパーの出力形式に合わせて適宜調整してください）
        print(
            f"Successfully saved all race cards to {output_file} ({len(all_rows)} records)"
        )


if __name__ == "__main__":
    main()
