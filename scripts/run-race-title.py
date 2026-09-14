#!/usr/bin/env python3
"""Scrape race title data from race.boatcast.jp (Standalone Version)"""

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
from typing import Optional

import requests

OUTPUT_DIR = "data/programs/title"
BOATCAST_BASE = "https://race.boatcast.jp"

STADIUM_NAMES = {
    "01": "桐生",
    "02": "戸田",
    "03": "江戸川",
    "04": "平和島",
    "05": "多摩川",
    "06": "浜名湖",
    "07": "蒲郡",
    "08": "常滑",
    "09": "津",
    "10": "三国",
    "11": "びわこ",
    "12": "住之江",
    "13": "尼崎",
    "14": "鳴門",
    "15": "丸亀",
    "16": "児島",
    "17": "宮島",
    "18": "徳山",
    "19": "下関",
    "20": "若松",
    "21": "芦屋",
    "22": "福岡",
    "23": "唐津",
    "24": "大村",
}

CSV_HEADER = [
    "レースコード",
    "レース日",
    "レース場コード",
    "レース場",
    "レース回",
    "タイトル",
    "日次",
    "グレード",
    "ナイター",
    "レース名",
    "電話投票締切予定",
    "中止状態",
]


def fetch_holding_list(date_str: str) -> Optional[dict]:
    yyyymmdd = date_str.replace("-", "")
    url = f"{BOATCAST_BASE}/api_txt/getHoldingList2_{yyyymmdd}.json"

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        ),
        "Accept": "application/json, text/plain, */*",
        "Referer": f"{BOATCAST_BASE}/",
    }

    try:
        resp = requests.get(url, headers=headers, timeout=30)
        if resp.status_code != 200:
            print(f"[WARN] HTTP Status {resp.status_code}: {url}")
            return None

        payload = resp.json()
        if not isinstance(payload, dict) or payload.get("res_cd") != 0:
            return None
        return payload
    except Exception as e:
        print(f"[ERROR] Fetch failed: {e}")
        return None


def _csv_escape(value: str) -> str:
    if value is None:
        return ""
    s = str(value)
    if any(c in s for c in (",", '"', "\n", "\r")):
        return '"' + s.replace('"', '""') + '"'
    return s


def build_csv(date_str: str, payload: dict) -> tuple[str, int]:
    yyyymmdd = date_str.replace("-", "")
    rows = []

    for venue in payload.get("return_info", []) or []:
        jo = str(venue.get("RaceStadiumNo") or "").zfill(2)
        if not jo or jo == "00":
            continue

        holding_title = (venue.get("HoldingTitle") or "").strip()
        daily_title = (venue.get("DailyTitle") or "").strip()
        race_grade = (venue.get("RaceGrade") or "").strip()
        nighter = (venue.get("NighterFlag") or "").strip()
        title_all = venue.get("RaceTitleAll") or []
        deadline_all = venue.get("DeadlineTimeAll") or []
        cancel_all = venue.get("CancelStatusAll") or []

        if not title_all and venue.get("RaceTitle"):
            title_all = [venue["RaceTitle"]]

        for idx, raw_title in enumerate(title_all, start=1):
            race_no = idx
            race_title = (raw_title or "").strip().strip("　").strip()

            deadline = (
                (deadline_all[idx - 1] or "").strip()
                if idx - 1 < len(deadline_all)
                else ""
            )
            cancel = (
                (cancel_all[idx - 1] or "").strip()
                if idx - 1 < len(cancel_all)
                else ""
            )

            race_code = f"{yyyymmdd}{jo}{race_no:02d}"
            stadium_name = STADIUM_NAMES.get(jo, "")

            rows.append(
                [
                    race_code,
                    date_str,
                    jo,
                    stadium_name,
                    f"{race_no}R",
                    holding_title,
                    daily_title,
                    race_grade,
                    nighter,
                    race_title,
                    deadline,
                    cancel,
                ]
            )

    if not rows:
        return "", 0

    rows.sort(key=lambda r: (r[2], int(r[4].rstrip("R"))))

    out_lines = [",".join(_csv_escape(c) for c in CSV_HEADER)]
    out_lines.extend(",".join(_csv_escape(c) for c in row) for row in rows)
    return "\n".join(out_lines) + "\n", len(rows)


def main():
    jst = timezone(timedelta(hours=9))
    today_jst = datetime.now(jst).strftime("%Y-%m-%d")

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default=today_jst)
    args = parser.parse_args()

    payload = fetch_holding_list(args.date)
    if not payload:
        print(f"No data fetched for {args.date}")
        sys.exit(0)

    csv_content, row_count = build_csv(args.date, payload)
    if row_count == 0:
        print(f"No races found for {args.date}")
        sys.exit(0)

    year, month, day = args.date.split("-")
    output_path = Path(f"{OUTPUT_DIR}/{year}/{month}/{day}.csv")

    # ディレクトリがなければ自動作成して保存
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(csv_content)

    print(f"Successfully saved {row_count} rows to {output_path}")


if __name__ == "__main__":
    main()
