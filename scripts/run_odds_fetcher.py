import csv
from datetime import datetime
import os
import sys

import pandas as pd

# scriptsフォルダをモジュール検索パスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from boatrace.downloader import RateLimiter
from boatrace.odds_realtime import ODDS_HEADERS, OddsRealtimeFetcher, build_odds_row


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

    # 時刻比較用のフル日時を作成
    df["close_datetime"] = pd.to_datetime(
        today_str + " " + df["電話投票締切予定"], format="%Y-%m-%d %H:%M"
    )

    # 現在時刻以降のレースを抽出して締め切り順にソート
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
    today_str = datetime.now().strftime("%Y-%m-%d")
    year, month, _ = today_str.split("-")

    # サーバー負荷軽減のためのウェイト
    fetcher = OddsRealtimeFetcher(rate_limiter=RateLimiter(interval_seconds=1.0))
    sources = ["od2", "od3"]

    # 直近5レースのみを取得対象に指定
    target_races = get_target_races(limit=5)
    print(f"Target races count: {len(target_races)}")

    for target in target_races:
        stadium_code = target["stadium_code"]
        race_number = target["race_number"]

        for source in sources:
            try:
                values = fetcher.fetch_values(
                    source=source,
                    date_str=today_str,
                    stadium_code=stadium_code,
                    race_number=race_number,
                )

                if values:
                    output_dir = f"data/previews/{year}/{month}"
                    os.makedirs(output_dir, exist_ok=True)
                    output_file = f"{output_dir}/{today_str.replace('-', '')}_{stadium_code:02d}_{race_number:02d}_{source}.csv"

                    headers = ODDS_HEADERS[source]
                    row = build_odds_row(
                        race_code=f"{today_str.replace('-', '')}_{stadium_code:02d}_{race_number:02d}",
                        date_str=today_str,
                        stadium_code=stadium_code,
                        race_number=race_number,
                        deadline_time=target["close_time"],
                        fetched_at_iso=datetime.now().isoformat(),
                        values=values,
                    )

                    with open(
                        output_file, "w", encoding="utf-8", newline=""
                    ) as f:
                        writer = csv.writer(f)
                        writer.writerow(headers)
                        writer.writerow(row)

            except Exception as e:
                print(
                    f"Error processing {stadium_code}R{race_number} ({source}): {e}"
                )
                pass


if __name__ == "__main__":
    main()
