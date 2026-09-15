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


def transform_to_wide(df_raw, stadium_code, race_number, deadline_time, today_str, now_jst_str):
    """画像のヘッダー順（1レース1行の横持ちフォーマット）に整形"""
    if df_raw is None or df_raw.empty:
        return None

    # すでに横持ち形式の場合はそのまま使用
    if "艇1_選手名" in df_raw.columns:
        df_wide = df_raw.copy()
    else:
        # 縦持ち（艇ごとの複数行）からの組み換え処理
        race_code = f"{today_str.replace('-', '')}{stadium_code:02d}{race_number:02d}"

        row = {
            "レースコード": race_code,
            "レース日": today_str,
            "レース場": f"{stadium_code:02d}",
            "レース回": f"{race_number:02d}R",
            "締切時刻": deadline_time,
            "取得日時": now_jst_str,
            "計測器": df_raw["計測器"].iloc[0] if "計測器" in df_raw.columns else "3",
            "計測項目1": df_raw["計測項目1"].iloc[0] if "計測項目1" in df_raw.columns else "一周",
            "計測項目2": df_raw["計測項目2"].iloc[0] if "計測項目2" in df_raw.columns else "まわり足",
            "計測項目3": df_raw["計測項目3"].iloc[0] if "計測項目3" in df_raw.columns else "直線",
        }

        # 艇番列の抽出
        boat_col = None
        for col in ["艇番", "艇", "pit_number", "boat_number"]:
            if col in df_raw.columns:
                boat_col = col
                break

        for i in range(1, 7):
            if boat_col:
                sub = df_raw[df_raw[boat_col].astype(str) == str(i)]
            else:
                sub = df_raw.iloc[i - 1 : i] if len(df_raw) >= i else pd.DataFrame()

            if not sub.empty:
                r = sub.iloc[0]
                name_val = r.get("選手名", r.get("racer_name", r.get("player_name", "")))
                val1 = r.get("値1", r.get("周回タイム", r.get("一周タイム", r.get("exhibition_time", ""))))
                val2 = r.get("値2", r.get("まわり足タイム", r.get("turn_time", "")))
                val3 = r.get("値3", r.get("直線タイム", r.get("straight_time", "")))
            else:
                name_val, val1, val2, val3 = "", "", "", ""

            row[f"艇{i}_選手名"] = name_val
            row[f"艇{i}_値1"] = val1
            row[f"艇{i}_値2"] = val2
            row[f"艇{i}_値3"] = val3

        df_wide = pd.DataFrame([row])

    # 画像通りの完全な列順を維持
    expected_cols = [
        "レースコード", "レース日", "レース場", "レース回", "締切時刻", "取得日時",
        "計測器", "計測項目1", "計測項目2", "計測項目3"
    ]
    for i in range(1, 7):
        expected_cols.extend([f"艇{i}_選手名", f"艇{i}_値1", f"艇{i}_値2", f"艇{i}_値3"])

    for col in expected_cols:
        if col not in df_wide.columns:
            df_wide[col] = ""

    return df_wide[expected_cols]


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

    # 時刻表記（例: 19:40）のみ抽出
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
            df_old = pd.read_csv(output_file, dtype=str)
            df_combined = pd.concat([df_old, df_new], ignore_index=True)
            
            # レースコード等で重複排除して最新データを残す
            dedup_cols = [c for c in ["レースコード", "race_code"] if c in df_combined.columns]
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

            df_raw = convert_to_dataframe(data)

            if df_raw is not None and not df_raw.empty:
                # 画像通りの横持ちフォーマット（1レース1行）に変換
                df_wide = transform_to_wide(
                    df_raw=df_raw,
                    stadium_code=stadium_code,
                    race_number=race_number,
                    deadline_time=deadline_time,
                    today_str=today_str,
                    now_jst_str=now_jst.isoformat(),
                )

                output_dir = f"data/previews/original_exhibition/{year}/{month}"
                os.makedirs(output_dir, exist_ok=True)
                output_file = f"{output_dir}/{day}.csv"

                save_or_update_csv(df_wide, output_file)

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

