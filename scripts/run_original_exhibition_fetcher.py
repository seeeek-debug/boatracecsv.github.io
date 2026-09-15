#!/usr/bin/env python3
"""
Fetch real-time original exhibition data for today's races
from race.boatcast.jp TSV files and save to
data/previews/original_exhibition/YYYY/MM/DD.csv
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
import os
import re
from pathlib import Path
import sys

import pandas as pd
import requests
from bs4 import BeautifulSoup

JST = timezone(timedelta(hours=9))

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

TARGET_COLUMNS = [
    "レースコード",
    "レース日",
    "レース場",
    "レース回",
    "締切時刻",
    "取得日時",
    "計測器",
    "計測項目1",
    "計測項目2",
    "計測項目3",
]

for i in range(1, 7):
    TARGET_COLUMNS.extend([
        f"艇{i}_選手名",
        f"艇{i}_値1",
        f"艇{i}_値2",
        f"艇{i}_値3",
    ])


def fetch_deadline_time(stadium_code: int, race_number: int, date_formatted: str) -> str:
    url = f"https://www.boatrace.jp/owpc/pc/race/beforeinfo?rno={race_number}&jcd={stadium_code:02d}&hd={date_formatted}"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    try:
        resp = requests.get(url, headers=headers, timeout=4)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.content, "html.parser")
            deadline_el = soup.select_one(".tab2_time, .label2, .is-deadline")
            if deadline_el:
                m = re.search(r'(\d{1,2}:\d{2})', deadline_el.text)
                if m:
                    return m.group(1)
    except Exception:
        pass
    return ""


def parse_ori_tsv(body_str: str) -> tuple[str, str, str, str, dict]:
    lines = body_str.splitlines()
    if not lines or not lines[0].strip().startswith("data="):
        return "3", "一周", "まわり足", "直線", {}

    meta_cols = lines[1].split("\t") if len(lines) > 1 else []
    device_id = meta_cols[0].strip() if len(meta_cols) > 0 and meta_cols[0].strip() else "3"
    item1 = meta_cols[1].strip() if len(meta_cols) > 1 and meta_cols[1].strip() else "一周"
    item2 = meta_cols[2].strip() if len(meta_cols) > 2 and meta_cols[2].strip() else "まわり足"
    item3 = meta_cols[3].strip() if len(meta_cols) > 3 and meta_cols[3].strip() else "直線"

    boats = {}
    boat_rows = []
    for raw in lines[2:]:
        if not raw.strip():
            continue
        boat_rows.append(raw)
        if len(boat_rows) >= 6:
            break

    for boat_num, raw in enumerate(boat_rows, start=1):
        cols = raw.split("\t")
        if len(cols) < 2:
            continue

        racer_name = cols[1].strip() if len(cols) > 1 else ""

        def parse_val(val_str):
            if not val_str or not val_str.strip():
                return ""
            try:
                return float(val_str.strip())
            except ValueError:
                return val_str.strip()

        v1 = parse_val(cols[2]) if len(cols) > 2 else ""
        v2 = parse_val(cols[3]) if len(cols) > 3 else ""
        v3 = parse_val(cols[4]) if len(cols) > 4 else ""

        boats[boat_num] = {
            "name": racer_name,
            "v1": v1,
            "v2": v2,
            "v3": v3,
        }

    return device_id, item1, item2, item3, boats


def fetch_ori_for_race(args) -> dict | None:
    stadium_code, race_number, date_str, now_jst = args
    date_formatted = date_str.replace("-", "")

    url = f"https://race.boatcast.jp/hp_txt/{stadium_code:02d}/bc_j_ori_{date_formatted}_{stadium_code:02d}_{race_number:02d}.txt"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    try:
        resp = requests.get(url, headers=headers, timeout=5)
        if resp.status_code != 200 or not resp.text.strip():
            return None

        device_id, item1, item2, item3, boats = parse_ori_tsv(resp.text)
        if not boats:
            return None

        deadline_time = fetch_deadline_time(stadium_code, race_number, date_formatted)
        race_code = f"{date_formatted}{stadium_code:02d}{race_number:02d}"

        row = {
            "レースコード": race_code,
            "レース日": date_str,
            "レース場": f"{stadium_code:02d}",
            "レース回": f"{race_number:02d}R",
            "締切時刻": deadline_time,
            "取得日時": now_jst.isoformat(),
            "計測器": device_id,
            "計測項目1": item1,
            "計測項目2": item2,
            "計測項目3": item3,
        }

        for i in range(1, 7):
            b = boats.get(i, {})
            row[f"艇{i}_選手名"] = b.get("name", "")
            row[f"艇{i}_値1"] = b.get("v1", "")
            row[f"艇{i}_値2"] = b.get("v2", "")
            row[f"艇{i}_値3"] = b.get("v3", "")

        return row

    except Exception:
        return None


def main():
    now_jst = datetime.now(JST)
    today_str = now_jst.strftime("%Y-%m-%d")
    year, month, day = now_jst.strftime("%Y"), now_jst.strftime("%m"), now_jst.strftime("%d")

    print(f"=== [START] Original Exhibition Fetcher: {today_str} (JST) ===", flush=True)

    output_dir = project_root / f"data/previews/original_exhibition/{year}/{month}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{day}.csv"

    existing_df = pd.DataFrame()
    if output_file.exists():
        try:
            existing_df = pd.read_csv(output_file, dtype=str)
        except Exception:
            existing_df = pd.DataFrame()

    tasks = [
        (stadium_code, race_number, today_str, now_jst)
        for stadium_code in range(1, 25)
        for race_number in range(1, 13)
    ]

    fetched_rows = []
    print(f"Fetching {len(tasks)} races in parallel...", flush=True)

    with ThreadPoolExecutor(max_workers=15) as executor:
        futures = [executor.submit(fetch_ori_for_race, task) for task in tasks]
        for future in as_completed(futures):
            res = future.result()
            if res:
                fetched_rows.append(res)

    if not fetched_rows:
        print(f"=== [NO DATA] 新規取得データなし ({today_str}) ===", flush=True)
        return

    new_df = pd.DataFrame(fetched_rows)

    if not existing_df.empty and "レースコード" in existing_df.columns:
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=["レースコード"], keep="last")
    else:
        combined_df = new_df

    combined_df = combined_df.sort_values(by=["レースコード"]).reset_index(drop=True)

    for col in TARGET_COLUMNS:
        if col not in combined_df.columns:
            combined_df[col] = ""
    combined_df = combined_df[TARGET_COLUMNS]

    combined_df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"=== [SUCCESS] Total {len(combined_df)} races saved -> {output_file} ===", flush=True)


if __name__ == "__main__":
    main()

