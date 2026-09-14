from datetime import datetime
import os
import csv
import sys

# scriptsフォルダをモジュール検索パスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from boatrace.odds_realtime import OddsRealtimeFetcher, build_odds_row, ODDS_HEADERS
from boatrace.downloader import RateLimiter

def main():
    today_str = datetime.now().strftime("%Y-%m-%d")
    year, month, _ = today_str.split("-")
    
    # サーバー負荷軽減のためのウェイト（1秒）
    fetcher = OddsRealtimeFetcher(rate_limiter=RateLimiter(interval_seconds=1.0))
    
    stadium_codes = range(1, 25) # 全24場
    sources = ["od2", "od3"] # od2（2連単・2連複など）, od3（3連単）
    
    for stadium_code in stadium_codes:
        for race_number in range(1, 13): # 1R〜12R
            for source in sources:
                try:
                    # オッズデータの取得とパース
                    values = fetcher.fetch_values(
                        source=source,
                        date_str=today_str,
                        stadium_code=stadium_code,
                        race_number=race_number
                    )
                    
                    if values:
                        # 年・月ごとにフォルダを階層化（例: data/previews/2026/09/）
                        output_dir = f"data/previews/{year}/{month}"
                        os.makedirs(output_dir, exist_ok=True)
                        output_file = f"{output_dir}/{today_str.replace('-', '')}_{stadium_code:02d}_{race_number:02d}_{source}.csv"
                        
                        headers = ODDS_HEADERS[source]
                        row = build_odds_row(
                            race_code=f"{today_str.replace('-', '')}_{stadium_code:02d}_{race_number:02d}",
                            date_str=today_str,
                            stadium_code=stadium_code,
                            race_number=race_number,
                            deadline_time="",
                            fetched_at_iso=datetime.now().isoformat(),
                            values=values
                        )
                        
                        # CSVファイルとして保存
                        with open(output_file, "w", encoding="utf-8", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow(headers)
                            writer.writerow(row)
                            
                except Exception:
                    # レースが開催されていない場合などはスキップ
                    pass

if __name__ == "__main__":
    main()

