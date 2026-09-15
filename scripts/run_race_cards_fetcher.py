from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone, timedelta
import os
from pathlib import Path
import sys

import pandas as pd
import numpy as np

# JST設定
JST = timezone(timedelta(hours=9))

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from boatrace.downloader import RateLimiter

try:
    from boatrace.race_card_scraper import RaceCardScraper
except ModuleNotFoundError:
    try:
        from boatrace.race_card import RaceCardScraper
    except ModuleNotFoundError:
        from boatrace.official.race_card import RaceCardFetcher as RaceCardScraper


def convert_to_df(data):
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
        return pd.DataFrame([
            asdict(x) if is_dataclass(x) else (vars(x) if hasattr(x, "__dict__") else x)
            for x in data
        ])
    return pd.DataFrame([data])


def safe_val(v):
    if v is None:
        return ""
    if isinstance(v, (list, tuple, np.ndarray, pd.Series)):
        if len(v) == 0:
            return ""
        v = v[0]
    try:
        if pd.isna(v):
            return ""
    except Exception:
        pass
    return v


def transform_wide(df_raw, stadium_code, race_number, today_str):
    """02.csv通りの横持ちフォーマット（レース情報 + 艇1〜艇6の列展開）に整形"""
    if df_raw is None or df_raw.empty:
        return None

    df_raw = df_raw.loc[:, ~df_raw.columns.duplicated()]

    stadium_str = f"{int(stadium_code):02d}"
    race_str = f"{int(race_number):02d}R"
    race_code = f"{today_str.replace('-', '')}{stadium_str}{int(race_number):02d}"

    row = {
        "レースコード": race_code,
        "レース日": today_str,
        "レース場コード": stadium_str,
        "レース回": race_str,
    }

    # すでに艇1_〜の横持ち形式になっている場合
    if any(str(col).startswith("艇1_") for col in df_raw.columns):
        first_row = df_raw.iloc[0].to_dict()
        for k, v in first_row.items():
            if k not in row and k is not None:
                row[k] = safe_val(v)
        return pd.DataFrame([row])

    # 縦持ち（1艇1行）の場合、1〜6艇のデータを横持ち変換
    boat_col = next((c for c in ["艇番", "艇", "pit_number", "boat_number"] if c in df_raw.columns), None)
    ignore_cols = {"レースコード", "レース日", "レース場コード", "レース場", "レース回", "レース", "stadium_code", "race_number", "date", boat_col}

    for i in range(1, 7):
        sub = df_raw[df_raw[boat_col].astype(str) == str(i)] if boat_col else (df_raw.iloc[i - 1 : i] if len(df_raw) >= i else pd.DataFrame())
        if not sub.empty:
            record = sub.iloc[0].to_dict()
            for k, v in record.items():
                if k not in ignore_cols and k is not None:
                    row[f"艇{i}_{k}"] = safe_val(v)

    return pd.DataFrame([row])


WORKING_CONFIG = {"method": None, "date": None, "is_kw": True}

def fetch_single_race(scraper, date_obj, date_str, stadium_code, race_number):
    global WORKING_CONFIG

    if WORKING_CONFIG["method"] is not None:
        try:
            m = WORKING_CONFIG["method"]
            d = WORKING_CONFIG["date"]
            return m(date=d, stadium_code=stadium_code, race_number=race_number) if WORKING_CONFIG["is_kw"] else m(d, stadium_code, race_number)
        except Exception:
            return None

    methods = ["scrape_race", "fetch_values", "scrape", "fetch"]
    dates = [date_obj, date_str, date_str.replace("-", "")]

    for m_name in methods:
        if hasattr(scraper, m_name):
            method = getattr(scraper, m_name)
            for d in dates:
                try:
                    res = method(date=d, stadium_code=stadium_code, race_number=race_number)
                    if res:
                        WORKING_CONFIG = {"method": method, "date": d, "is_kw": True}
                        return res
                except Exception:
                    pass
                try:
                    res = method(d, stadium_code, race_number)
                    if res:
                        WORKING_CONFIG = {"method": method, "date": d, "is_kw": False}
                        return res
                except Exception:
                    pass
    return None


def main():
    now_jst = datetime.now(JST)
    today_date_obj = now_jst.date()
    today_str = now_jst.strftime("%Y-%m-%d")
    year, month, day = now_jst.strftime("%Y"), now_jst.strftime("%m"), now_jst.strftime("%d")

    print(f"=== [START] Target Date: {today_str} (JST) ===", flush=True)

    scraper = RaceCardScraper(rate_limiter=RateLimiter(interval_seconds=0.2))
    all_dfs = []

    for stadium_code in range(1, 25):
        stadium_success = 0
        for race_number in range(1, 13):
            data = fetch_single_race(scraper, today_date_obj, today_str, stadium_code, race_number)
            if data is not None:
                df_raw = convert_to_df(data)
                df_wide = transform_wide(df_raw, stadium_code, race_number, today_str)
                if df_wide is not None and not df_wide.empty:
                    all_dfs.append(df_wide)
                    stadium_success += 1

        if stadium_success > 0:
            print(f"  [場コード {stadium_code:02d}] {stadium_success} レース取得完了", flush=True)

    if all_dfs:
        output_dir = f"data/programs/race_cards/{year}/{month}"
        os.makedirs(output_dir, exist_ok=True)
        output_file = f"{output_dir}/{day}.csv"

        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_df.to_csv(output_file, index=False, encoding="utf-8-sig")
        print(f"=== [SUCCESS] Saved {len(combined_df)} races -> {output_file} ===", flush=True)
    else:
        print(f"=== [NO DATA] No races found for {today_str} ===", flush=True)


if __name__ == "__main__":
    main()

