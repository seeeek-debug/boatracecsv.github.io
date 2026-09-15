#!/usr/bin/env python3
"""
Fetch real-time sui (weather & water condition) preview data for today's races
and update data/previews/sui/YYYY/MM/DD.csv to match the official preview schema.
"""
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

WEATHER_MAP = {
    "晴": 1, "曇": 2, "雨": 3, "雪": 4, "霧": 5
}

WIND_DIR_MAP = {
    "北": 1, "北北東": 2, "北東": 2, "東北東": 3,
    "東": 3, "東南東": 4, "南東": 4, "南南東": 5,
    "南": 5, "南南西": 6, "南西": 6, "西南西": 7,
    "西": 7, "西北西": 8, "北西": 8, "北北西": 1
}

# 画像（06/18.csv）と完全に一致する13列の定義
TARGET_COLUMNS = [
    "レースコード",
    "レース日",
    "レース場",
    "レース回",
    "締切時刻",
    "取得日時",
    "気象観測時刻",
    "風速(m)",
    "風向",
    "波の高さ(cm)",
    "天候",
    "気温(℃)",
    "水温(℃)",
]


def parse_wind_dir(val_str: str) -> str:
    """風向テキストまたはクラス名から1~8のコード文字列に変換（該当なしは空文字）"""
    if not val_str:
        return ""
    m = re.search(r'\d+', val_str)
    if m:
        num = int(m.group())
        if 1 <= num <= 16:
            deg_map = {1:1, 2:2, 3:2, 4:3, 5:3, 6:4, 7:4, 8:5, 9:5, 10:6, 11:6, 12:7, 13:7, 14:8, 15:8, 16:1}
            return str(deg_map.get(num, ""))
    for k, v in WIND_DIR_MAP.items():
        if k in val_str:
            return str(v)
    return ""


def fetch_sui_for_race(stadium_code: int, race_number: int, date_str: str, now_jst: datetime) -> dict | None:
    """ボートレース公式サイトの直前情報ページから全気象項目を取得"""
    date_formatted = date_str.replace("-", "")
    url = f"https://www.boatrace.jp/owpc/pc/race/beforeinfo?rno={race_number}&jcd={stadium_code:02d}&hd={date_formatted}"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code != 200:
            return None

        soup = BeautifulSoup(resp.content, "html.parser")
        weather_box = soup.select_one(".weather1")
        if not weather_box:
            return None

        # 締切時刻の取得
        deadline_time = ""
        deadline_el = soup.select_one(".tab2_time, .label2, .is-deadline")
        if deadline_el:
            m_dl = re.search(r'(\d{1,2}:\d{2})', deadline_el.text)
            if m_dl:
                deadline_time = m_dl.group(1)

        # 気象観測時刻の取得（例: 0755）
        obs_time = ""
        obs_el = weather_box.select_one(".weather1_title, .weather1_time, .weather1_bodyTime")
        if obs_el:
            m_obs = re.search(r'(\d{1,2})[:：]?(\d{2})', obs_el.text)
            if m_obs:
                obs_time = f"{int(m_obs.group(1)):02d}{m_obs.group(2)}"

        # 気温・天候・風速・風向・水温・波高の解析
        temp_el = weather_box.select_one(".weather1_bodyUnit--sora .weather1_bodyUnitLabelData")
        air_temp = float(re.search(r'[\d\.]+', temp_el.text).group()) if temp_el and re.search(r'[\d\.]+', temp_el.text) else None

        weather_el = weather_box.select_one(".weather1_bodyUnit--sora .weather1_bodyUnitLabelTitle")
        weather_text = weather_el.text.strip() if weather_el else ""
        weather_code = WEATHER_MAP.get(weather_text, 1)

        wind_ms_el = weather_box.select_one(".weather1_bodyUnit--kaze .weather1_bodyUnitLabelData")
        wind_ms = float(re.search(r'[\d\.]+', wind_ms_el.text).group()) if wind_ms_el and re.search(r'[\d\.]+', wind_ms_el.text) else 0.0

        wind_dir_code = ""
        if wind_ms > 0:
            wind_dir_el = weather_box.select_one(".weather1_bodyUnit--kaze .weather1_bodyUnitImage")
            wind_dir_class = wind_dir_el.get("class", []) if wind_dir_el else []
            wind_dir_code = parse_wind_dir(" ".join(wind_dir_class))

        water_temp_el = weather_box.select_one(".weather1_bodyUnit--mizu .weather1_bodyUnitLabelData")
        water_temp = float(re.search(r'[\d\.]+', water_temp_el.text).group()) if water_temp_el and re.search(r'[\d\.]+', water_temp_el.text) else None

        wave_el = weather_box.select_one(".weather1_bodyUnit--nami .weather1_bodyUnitLabelData")
        wave_cm = float(re.search(r'[\d\.]+', wave_el.text).group()) if wave_el and re.search(r'[\d\.]+', wave_el.text) else 0.0

        if air_temp is None or water_temp is None:
            return None

        race_code = f"{date_formatted}{stadium_code:02d}{race_number:02d}"

        return {
            "レースコード": race_code,
            "レース日": date_str,
            "レース場": f"{stadium_code:02d}",
            "レース回": f"{race_number:02d}R",
            "締切時刻": deadline_time,
            "取得日時": now_jst.isoformat(),
            "気象観測時刻": obs_time,
            "風速(m)": wind_ms,
            "風向": wind_dir_code,
            "波の高さ(cm)": wave_cm,
            "天候": weather_code,
            "気温(℃)": air_temp,
            "水温(℃)": water_temp,
        }

    except Exception:
        return None


def main():
    now_jst = datetime.now(JST)
    today_str = now_jst.strftime("%Y-%m-%d")
    year, month, day = now_jst.strftime("%Y"), now_jst.strftime("%m"), now_jst.strftime("%d")

    print(f"=== [START] Sui Fetcher: {today_str} (JST) ===", flush=True)

    output_dir = project_root / f"data/previews/sui/{year}/{month}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{day}.csv"

    existing_df = pd.DataFrame()
    if output_file.exists():
        try:
            existing_df = pd.read_csv(output_file, dtype=str)
        except Exception:
            existing_df = pd.DataFrame()

    fetched_rows = []
    for stadium_code in range(1, 25):
        stadium_success = 0
        for race_number in range(1, 13):
            data = fetch_sui_for_race(stadium_code, race_number, today_str, now_jst)
            if data:
                fetched_rows.append(data)
                stadium_success += 1

        if stadium_success > 0:
            print(f"  [場コード {stadium_code:02d}] {stadium_success} レースの気象データ取得完了", flush=True)

    if not fetched_rows:
        print(f"=== [NO DATA] 新規取得データなし ({today_str}) ===", flush=True)
        return

    new_df = pd.DataFrame(fetched_rows)

    if not existing_df.empty and "レースコード" in existing_df.columns:
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=["レースコード"], keep="last")
    else:
        combined_df = new_df

    # 定義した13列の順番に並び替えて保存
    for col in TARGET_COLUMNS:
        if col not in combined_df.columns:
            combined_df[col] = ""
    combined_df = combined_df[TARGET_COLUMNS]

    combined_df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"=== [SUCCESS] Total {len(combined_df)} races saved -> {output_file} ===", flush=True)


if __name__ == "__main__":
    main()

