from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone, timedelta
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

# インポートの柔軟対応
try:
    from boatrace.race_card_scraper import RaceCardScraper
except ModuleNotFoundError:
    try:
        from boatrace.race_card import RaceCardScraper
    except ModuleNotFoundError:
        from boatrace.official.race_card import RaceCardFetcher as RaceCardScraper


def convert_to_dataframe(data):
    """カスタムオブジェクト/辞書/リストを DataFrame に安全変換"""
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


def transform_to_wide(df_raw, stadium_code, race_number, today_str):
    """取得した出走表データを画像通りの横持ちフォーマット（1レース1行）に変換"""
    if df_raw is None or df_raw.empty:
        return None

    # すでに横持ち形式（艇1_で始まる列が存在する）の場合はそのまま返す
    if any(str(col).startswith("艇1_") for col in df_raw.columns):
        return df_raw

    # レース基本情報
    stadium_str = f"{int(stadium_code):02d}"
    race_str = f"{int(race_number):02d}R"
    race_code = f"{today_str.replace('-', '')}{stadium_str}{int(race_number):02d}"

    row = {
        "レースコード": race_code,
        "レース日": today_str,
        "レース場コード": stadium_str,
        "レース回": race_str,
    }

    # 艇番を表す列名を自動特定
    boat_col = None
    for c in ["艇番", "艇", "pit_number", "boat_number"]:
        if c in df_raw.columns:
            boat_col = c
            break

    # 共通ヘッダーとして除外するカラム群
    ignore_cols = {
        "レースコード", "レース日", "レース場コード", "レース場", "レース回",
        "レース", "stadium_code", "race_number", "date", boat_col
    }

    # 1艇〜6艇のデータを横に結合
    for i in range(1, 7):
        if boat_col:
            sub = df_raw[df_raw[boat_col].astype(str) == str(i)]
        else:
            sub = df_raw.iloc[i - 1 : i] if len(df_raw) >= i else pd.DataFrame()

        if not sub.empty:
            r = sub.iloc[0].to_dict()
            for k, v in r.items():
                if k in ignore_cols or k is None:
                    continue
                row[f"艇{i}_{k}"] = "" if pd.isna(v) else v

    return pd.DataFrame([row])


def save_or_update_csv(df_new, output_file):
    """既存CSVがある場合は最新データに結合・重複排除して保存"""
    if os.path.exists(output_file):
        try:
            df_old = pd.read_csv(output_file, dtype=str)
            df_combined = pd.concat([df_old, df_new], ignore_index=True)

            dedup_cols = [c for c in ["レースコード", "race_code"] if c in df_combined.columns]
            if dedup_cols:
                df_combined = df_combined.drop_duplicates(subset=dedup_cols, keep="last")
            else:
                df_combined = df_combined.drop_duplicates(keep="last")

            df_combined.to_csv(output_file, index=False, encoding="utf-8-sig")
            return
        except Exception as e:
            print(f"[Warning] CSVのマージに失敗したため新規作成します: {e}")

    df_new.to_csv(output_file, index=False, encoding="utf-8-sig")


def main():
    now_jst = datetime.now(JST)
    today_str = now_jst.strftime("%Y-%m-%d")
    year, month, day = now_jst.strftime("%Y"), now_jst.strftime("%m"), now_jst.strftime("%d")

    scraper = RaceCardScraper(rate_limiter=RateLimiter(interval_seconds=1.0))
    stadium_codes = range(1, 25)  # 全24場

    print(f"Start fetching and transforming all race cards for {today_str} (JST)")

    all_dfs = []
    for stadium_code in stadium_codes:
        for race_number in range(1, 13):  # 1R 〜 12R
            try:
                # メソッド名の自動判別呼び出し
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
                elif hasattr(scraper, "scrape"):
                    data = scraper.scrape(
                        date=today_str,
                        stadium_code=stadium_code,
                        race_number=race_number,
                    )
                else:
                    data = None

                df_raw = convert_to_dataframe(data)

                if df_raw is not None and not df_raw.empty:
                    # 1レース1行の横持ちフォーマットに変換
                    df_wide = transform_to_wide(
                        df_raw=df_raw,
                        stadium_code=stadium_code,
                        race_number=race_number,
                        today_str=today_str,
                    )
                    if df_wide is not None and not df_wide.empty:
                        all_dfs.append(df_wide)

            except Exception:
                # 非開催場・未開催レースのスキップ
                pass

    # CSVファイルへの書き出し
    if all_dfs:
        output_dir = f"data/programs/race_cards/{year}/{month}"
        os.makedirs(output_dir, exist_ok=True)
        output_file = f"{output_dir}/{day}.csv"

        combined_df = pd.concat(all_dfs, ignore_index=True)
        save_or_update_csv(combined_df, output_file)

        print(
            f"Successfully saved all race cards to {output_file} ({len(combined_df)} races)"
        )
    else:
        print("No race card data found to save.")


if __name__ == "__main__":
    main()
