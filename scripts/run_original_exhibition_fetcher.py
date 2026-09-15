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
import boatrace.original_exhibition_scraper as ex_module

# --- クラス自動判定処理 ---
ScraperClass = None
for candidate in ["OriginalExhibitionScraper", "OriginalExhibition", "ExhibitionScraper"]:
    if hasattr(ex_module, candidate):
        ScraperClass = getattr(ex_module, candidate)
        break

if ScraperClass is None:
    classes = [
        obj for name, obj in inspect.getmembers(ex_module, inspect.isclass)
        if obj.__module__ == ex_module.__name__
    ]
    if classes:
        ScraperClass = classes[0]
    else:
        print("Error: boatrace/original_exhibition_scraper.py 内にクラスが見つかりません。")
        sys.exit(1)


def convert_to_dataframe(data):
    """OriginalExhibitionData などのカスタムオブジェクトを DataFrame に安全変換"""
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

    # 時刻表記（例: 19:40）のみを抽出（「締切」などの文字列を除去）
    df["clean_time"] = df["電話投票締切予定"].astype(str).str.extract(r"(\d{1,2}:\d{2})")[0]
    df = df.dropna(subset=["clean_time"])

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


def save_or_update_csv(df_new, output_file):
    """既存CSVがある場合は同じレースのデータを最新版へ上書き保存"""
    if os.path.exists(output_file):
        try:
            df_old = pd.read_csv(output_file)
            df_combined = pd.concat([df_old, df_new], ignore_index=True)
            
            # 重複判定キー候補（レースコード等）があれば最新（last）を残す
            dedup_cols = [c for c in ["レースコード", "race_code", "stadium_code", "race_number"] if c in df_combined.columns]
            if dedup_cols:
                df_combined = df_combined.drop_duplicates(subset=dedup_cols, keep="last")
            else:
                df_combined = df_combined.drop_duplicates(keep="last")

            df_combined.to_csv(output_file, index=False, encoding="utf-8-sig")
            return
        except Exception as e:
            print(f"[Warning] Merging CSV failed, writing directly: {e}")

    df_new.to_csv(output_file, index=False, encoding="utf-8-sig")


def main():
    now_jst = datetime.now(JST)
    today_str = now_jst.strftime("%Y-%m-%d")
    year, month, day = now_jst.strftime("%Y"), now_jst.strftime("%m"), now_jst.strftime("%d")

    scraper = ScraperClass(rate_limiter=RateLimiter(interval_seconds=1.0))
    target_races = get_target_races(now_jst, limit=3)

    print(f"Target original exhibition count: {len(target_races)}")

    if not target_races:
        print("対象レースが見つかりません（プログラムCSVが存在しないか全レース終了済み）。")
        return

    for target in target_races:
        stadium_code = target["stadium_code"]
        race_number = target["race_number"]
        deadline_time = target["close_time"]

        try:
            if hasattr(scraper, "scrape_race"):
                data = scraper.scrape_race(
                    date=today_str,
                    stadium_code=stadium_code,
                    race_number=race_number,
                )
            elif hasattr(scraper, "fetch_values"):
                data = scraper.fetch_values(
                    date=today_str,
                    stadium_code=stadium_code,
                    race_number=race_number,
                )
            else:
                print("Error: 適切なデータ取得メソッドが見つかりません。")
                break

            df_new = convert_to_dataframe(data)

            if df_new is not None and not df_new.empty:
                output_dir = f"data/previews/original_exhibition/{year}/{month}"
                os.makedirs(output_dir, exist_ok=True)
                output_file = f"{output_dir}/{day}.csv"

                save_or_update_csv(df_new, output_file)

                print(
                    f"Saved/Updated original exhibition data to {output_file} "
                    f"({stadium_code}R{race_number}, 締切予定:{deadline_time})"
                )

        except Exception as e:
            print(
                f"Error processing exhibition {stadium_code}R{race_number}: {e}"
            )


if __name__ == "__main__":
    main()

