from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone, timedelta
import os
from pathlib import Path
import sys

import pandas as pd
import numpy as np

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

KEY_MAP = {
    "registration_number": "登録番号", "racer_id": "登録番号", "toban": "登録番号", "登番": "登録番号", "登録番号": "登録番号",
    "racer_name": "選手名", "player_name": "選手名", "name": "選手名", "選手名": "選手名",
    "period": "期別", "term": "期別", "期別": "期別", "期": "期別",
    "branch": "支部", "支部": "支部",
    "birthplace": "出身地", "hometown": "出身地", "出身地": "出身地", "出身": "出身地",
    "age": "年齢", "年齢": "年齢",
    "class_rank": "級別", "grade": "級別", "rank": "級別", "級別": "級別", "級": "級別",
    "prize_exclusion": "賞除", "shojo": "賞除", "賞除": "賞除",
    "f_count": "F本数", "f_number": "F本数", "false_starts": "F本数", "f": "F本数", "F本数": "F本数",
    "l_count": "L本数", "l_number": "L本数", "late_starts": "L本数", "l": "L本数", "L本数": "L本数",
    "national_st_avg": "全国平均ST", "national_avg_st": "全国平均ST", "st_avg": "全国平均ST", "avg_st": "全国平均ST", "全国平均ST": "全国平均ST", "平均ST": "全国平均ST",
    "national_win_rate": "全国勝率", "win_rate": "全国勝率", "全国勝率": "全国勝率",
    "national_2in_rate": "全国2連対率", "national_2ren": "全国2連対率", "2in_rate": "全国2連対率", "全国2連対率": "全国2連対率",
    "national_3in_rate": "全国3連対率", "national_3ren": "全国3連対率", "3in_rate": "全国3連対率", "全国3連対率": "全国3連対率",
    "local_win_rate": "当地勝率", "当地勝率": "当地勝率",
    "local_2in_rate": "当地2連対率", "local_2ren": "当地2連対率", "当地2連対率": "当地2連対率",
    "local_3in_rate": "当地3連対率", "local_3ren": "当地3連対率", "当地3連対率": "当地3連対率",
    "motor_flag": "モーターフラグ", "モーターフラグ": "モーターフラグ",
    "motor_number": "モーター番号", "motor_no": "モーター番号", "motor_id": "モーター番号", "モーター番号": "モーター番号", "モーター": "モーター番号",
    "motor_2in_rate": "モーター2連対率", "motor_2ren": "モーター2連対率", "モーター2連対率": "モーター2連対率",
    "motor_3in_rate": "モーター3連対率", "motor_3ren": "モーター3連対率", "モーター3連対率": "モーター3連対率",
    "boat_flag": "ボートフラグ", "ボートフラグ": "ボートフラグ",
    "boat_number": "ボート番号", "boat_no": "ボート番号", "boat_id": "ボート番号", "ボート番号": "ボート番号", "ボート": "ボート番号",
    "boat_2in_rate": "ボート2連対率", "boat_2ren": "ボート2連対率", "ボート2連対率": "ボート2連対率",
    "boat_3in_rate": "ボート3連対率", "boat_3ren": "ボート3連対率", "ボート3連対率": "ボート3連対率",
    "hayami": "早見", "早見": "早見"
}

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

def to_python_object(obj):
    if obj is None:
        return None
    if is_dataclass(obj):
        return asdict(obj)
    if hasattr(obj, "__dict__"):
        res = {}
        for k, v in vars(obj).items():
            if is_dataclass(v):
                res[k] = asdict(v)
            elif isinstance(v, list):
                res[k] = [asdict(x) if is_dataclass(x) else (vars(x) if hasattr(x, "__dict__") else x) for x in v]
            elif hasattr(v, "__dict__"):
                res[k] = vars(v)
            else:
                res[k] = v
        return res
    if isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            if is_dataclass(v):
                res[k] = asdict(v)
            elif isinstance(v, list):
                res[k] = [asdict(x) if is_dataclass(x) else (vars(x) if hasattr(x, "__dict__") else x) for x in v]
            elif hasattr(v, "__dict__"):
                res[k] = vars(v)
            else:
                res[k] = v
        return res
    return obj

def flatten_dict(d, parent_key='', sep='_'):
    items = []
    if not isinstance(d, dict):
        return {}
    for k, v in d.items():
        k_str = str(k)
        mapped_k = KEY_MAP.get(k_str, k_str)
        new_key = f"{parent_key}{sep}{mapped_k}" if parent_key else mapped_k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            for idx, item in enumerate(v):
                if isinstance(item, dict):
                    items.extend(flatten_dict(item, f"{new_key}{idx+1}", sep=sep).items())
                else:
                    items.append((f"{new_key}_{idx+1}", item))
        else:
            items.append((new_key, v))
    return dict(items)

def transform_race_to_wide(data, stadium_code, race_number, today_str):
    if data is None:
        return None

    py_data = to_python_object(data)
    stadium_str = f"{int(stadium_code):02d}"
    race_str = f"{int(race_number):02d}R"
    race_code = f"{today_str.replace('-', '')}{stadium_str}{int(race_number):02d}"

    row = {
        "レースコード": race_code,
        "レース日": today_str,
        "レース場コード": stadium_str,
        "レース回": race_str,
    }

    boats_list = []
    if isinstance(py_data, list):
        boats_list = py_data
    elif isinstance(py_data, dict):
        for key in ["boats", "entries", "pit_cards", "racers", "runners", "boat_cards"]:
            if key in py_data and isinstance(py_data[key], list):
                boats_list = py_data[key]
                break
        if not boats_list:
            for k, v in py_data.items():
                if isinstance(v, list) and len(v) > 0 and isinstance(v[0], (dict, list)):
                    boats_list = v
                    break

    if not boats_list and isinstance(data, pd.DataFrame):
        df = data
        boat_col = next((c for c in ["艇番", "艇", "pit_number", "boat_number"] if c in df.columns), None)
        for i in range(1, 7):
            sub = df[df[boat_col].astype(str) == str(i)] if boat_col else (df.iloc[i - 1 : i] if len(df) >= i else pd.DataFrame())
            if not sub.empty:
                rec = sub.iloc[0].to_dict()
                rec_flat = flatten_dict(rec)
                for k, v in rec_flat.items():
                    if k not in ["レースコード", "レース日", "レース場コード", "レース回", boat_col]:
                        row[f"艇{i}_{k}"] = safe_val(v)
        return pd.DataFrame([row])

    for idx in range(6):
        i = idx + 1
        if idx < len(boats_list):
            b_raw = boats_list[idx]
            b_dict = to_python_object(b_raw)
            if isinstance(b_dict, dict):
                flat_b = flatten_dict(b_dict)
                for k, v in flat_b.items():
                    if k in ["boat_number", "pit_number", "lane_number", "艇番"] and str(v) == str(i):
                        continue
                    if k == "期別" and str(v).isdigit() and not str(v).endswith("期"):
                        v = f"{v}期"
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
                df_wide = transform_race_to_wide(data, stadium_code, race_number, today_str)
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

