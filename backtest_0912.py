import io
import itertools
import os
import time
from datetime import datetime, timedelta
import joblib
import numpy as np
import pandas as pd
import requests

GITHUB_RAW_BASE = (
    "https://raw.githubusercontent.com/seeeek-debug/boatracecsv.github.io/main/"
)
CSV_CACHE = {}
CACHE_TTL = 300

VENUE_MAPPING = {
    "桐生": "01",
    "戸田": "02",
    "江戸川": "03",
    "平和島": "04",
    "多摩川": "05",
    "浜名湖": "06",
    "蒲郡": "07",
    "常滑": "08",
    "津": "09",
    "三国": "10",
    "びわこ": "11",
    "住之江": "12",
    "尼崎": "13",
    "鳴門": "14",
    "丸亀": "15",
    "児島": "16",
    "宮島": "17",
    "徳山": "18",
    "下関": "19",
    "若松": "20",
    "芦屋": "21",
    "福岡": "22",
    "唐津": "23",
    "大村": "24",
}

VENUE_PREVIEW_CODE_MAP = {
    "01": "kir",
    "02": "tod",
    "03": "edg",
    "04": "hei",
    "05": "tam",
    "06": "ham",
    "07": "gam",
    "08": "tkz",
    "09": "tsu",
    "10": "mik",
    "11": "biw",
    "12": "sum",
    "13": "ama",
    "14": "nar",
    "15": "mar",
    "16": "koj",
    "17": "miy",
    "18": "tok",
    "19": "shm",
    "20": "wkm",
    "21": "ash",
    "22": "fuk",
    "23": "ktu",
    "24": "omr",
}


def clean_name(val):
    if pd.isna(val):
        return ""
    return str(val).replace(" ", "").replace("　", "").strip()


def fetch_github_csv(file_path, use_cache=True):
    now = time.time()
    if use_cache and file_path in CSV_CACHE:
        ts, cached_df = CSV_CACHE[file_path]
        if now - ts < CACHE_TTL:
            return cached_df

    timestamp = int(now)
    uri = f"{GITHUB_RAW_BASE}{file_path}?t={timestamp}"
    try:
        res = requests.get(uri, timeout=10)
        if res.status_code == 200:
            df = pd.read_csv(
                io.StringIO(res.text), encoding="utf-8-sig", dtype=str
            )
            df.columns = df.columns.str.strip()
            CSV_CACHE[file_path] = (now, df)
            return df
    except Exception:
        pass
    return None


def fetch_github_csv_with_fallback(primary_path, fallback_path, use_cache=True):
    df = fetch_github_csv(primary_path, use_cache=use_cache)
    if df is None and fallback_path:
        df = fetch_github_csv(fallback_path, use_cache=use_cache)
    return df


def get_season(m):
    if m in [3, 4, 5]:
        return "春"
    elif m in [6, 7, 8]:
        return "夏"
    elif m in [9, 10, 11]:
        return "秋"
    else:
        return "冬"


MODEL_FILENAME = "boatrace_lgb_model.pkl"
models = {}
player_fav_kimarite = None
kimarite_prob_dict = {}
expected_features = []
feature_medians = {}
cat_categories = {}

if os.path.exists(MODEL_FILENAME):
    loaded_package = joblib.load(MODEL_FILENAME)
    if isinstance(loaded_package, dict):
        if "model_1st" in loaded_package:
            models["rank_1"] = loaded_package.get("model_1st")
            models["rank_2"] = loaded_package.get("model_2nd")
            models["rank_3"] = loaded_package.get("model_3rd")
        elif "models" in loaded_package:
            models = loaded_package.get("models")
        elif "rank_1" in loaded_package:
            models = loaded_package

        player_fav_kimarite = loaded_package.get("player_fav_kimarite")
        loaded_pair_table = loaded_package.get("pair_table")
        if loaded_pair_table and isinstance(loaded_pair_table, dict):
            kimarite_prob_dict = loaded_pair_table

        if "feature_names" in loaded_package:
            expected_features = loaded_package["feature_names"]
        elif "features" in loaded_package:
            expected_features = loaded_package["features"]

        if "feature_medians" in loaded_package:
            feature_medians = loaded_package.get("feature_medians", {})
        if "cat_categories" in loaded_package:
            cat_categories = loaded_package.get("cat_categories", {})

if not kimarite_prob_dict:
    df_pair = fetch_github_csv(
        "data/estimate/kimarite/tables/pair_table.csv", use_cache=True
    )
    if df_pair is not None:
        for _, row in df_pair.iterrows():
            k_type = str(row["セル"]).strip()
            c2 = int(row["2着コース"])
            c3 = int(row["3着コース"])
            prob = float(row["確率"])
            kimarite_prob_dict[(k_type, c2, c3)] = prob


def predict_single_race(
    venue_code, year, month_str, day_str_zf, month_raw, day_raw, r_num, df_odds=None
):
    venue_s = str(venue_code).zfill(2)
    month_int = int(month_str)

    race_card_p1 = f"data/programs/race_cards/{year}/{month_str}/{day_str_zf}.csv"
    race_card_p2 = f"data/programs/race_cards/{year}/{month_raw}/{day_raw}.csv"
    sui_p1 = f"data/previews/sui/{year}/{month_str}/{day_str_zf}.csv"
    sui_p2 = f"data/previews/sui/{year}/{month_raw}/{day_raw}.csv"
    orig_p1 = f"data/previews/original_exhibition/{year}/{month_str}/{day_str_zf}.csv"
    orig_p2 = f"data/previews/original_exhibition/{year}/{month_raw}/{day_raw}.csv"
    stt_p1 = f"data/previews/stt/{year}/{month_str}/{day_str_zf}.csv"
    stt_p2 = f"data/previews/stt/{year}/{month_raw}/{day_raw}.csv"

    prev_code = VENUE_PREVIEW_CODE_MAP.get(venue_s, "")
    venue_preview_p1 = (
        f"data/previews/{prev_code}/{year}/{month_str}/{day_str_zf}.csv"
        if prev_code
        else None
    )
    venue_preview_p2 = (
        f"data/previews/{prev_code}/{year}/{month_raw}/{day_raw}.csv"
        if prev_code
        else None
    )

    df_cards = fetch_github_csv_with_fallback(race_card_p1, race_card_p2, use_cache=True)
    if df_cards is None:
        return None

    df_sui = fetch_github_csv_with_fallback(sui_p1, sui_p2, use_cache=True)
    df_orig = fetch_github_csv_with_fallback(orig_p1, orig_p2, use_cache=True)
    df_stt = fetch_github_csv_with_fallback(stt_p1, stt_p2, use_cache=True)
    df_venue_preview = fetch_github_csv_with_fallback(
        venue_preview_p1, venue_preview_p2, use_cache=True
    )

    df_course_win = fetch_github_csv(
        "data/estimate/stadium/course_win_rate.csv", use_cache=True
    )
    df_season_win = fetch_github_csv(
        "data/estimate/stadium/win_rate.csv", use_cache=True
    )

    r_str = str(r_num).zfill(2)
    target_race_code = f"{year}{month_str}{day_str_zf}{venue_s}{r_str}"

    def get_matched_row(df, code):
        if df is None:
            return None
        for col in df.columns:
            if "レースコード" in col or "code" in col.lower():
                col_vals = (
                    df[col].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
                )
                matched = df[col_vals == str(code)]
                if len(matched) > 0:
                    return matched.iloc[0].to_dict()
        return None

    card_row = get_matched_row(df_cards, target_race_code)
    if not card_row:
        return None

    combined_row = {}
    combined_row.update(card_row)
    combined_row.update(get_matched_row(df_sui, target_race_code) or {})
    combined_row.update(get_matched_row(df_orig, target_race_code) or {})
    combined_row.update(get_matched_row(df_stt, target_race_code) or {})
    combined_row.update(get_matched_row(df_venue_preview, target_race_code) or {})

    if df_course_win is not None:
        for _, row in df_course_win.iterrows():
            if str(row.get("場コード", "")).strip().zfill(2) == venue_s and str(
                row.get("レース回", "")
            ).strip() == str(int(r_num)):
                for k, v in row.items():
                    if k not in ["場コード", "レース回"]:
                        combined_row[f"est_course_{k}"] = v
                break

    if df_season_win is not None:
        season_name = get_season(month_int)
        for _, row in df_season_win.iterrows():
            if str(row.get("場コード", "")).strip().zfill(2) == venue_s and str(
                row.get("季節", "")
            ).strip() == season_name:
                for k, v in row.items():
                    if k not in ["場コード", "季節"]:
                        combined_row[f"est_season_{k}"] = v
                break

    df_pred = pd.DataFrame([combined_row])

    if player_fav_kimarite:
        dummy_k_keys = (
            list(next(iter(player_fav_kimarite.values())).keys())
            if player_fav_kimarite
            else []
        )
        for i in range(1, 7):
            p_col_candidates = [
                f"艇{i}_選手名",
                f"{i}号艇_選手名",
                f"選手名_{i}",
                f"艇{i}_氏名",
                f"{i}号艇_氏名",
                f"氏名_{i}",
                f"艇{i}_選手",
                f"{i}号艇_選手",
            ]
            p_val = ""
            for c in p_col_candidates:
                if c in df_pred.columns and pd.notna(df_pred.iloc[0][c]):
                    val = clean_name(df_pred.iloc[0][c])
                    if val and val != "nan":
                        p_val = val
                        break
            for k_name in dummy_k_keys:
                df_pred[f"艇{i}_kimarite_{k_name}"] = player_fav_kimarite.get(
                    p_val, {}
                ).get(k_name, 0.0)

    X_input = df_pred.reindex(columns=expected_features)
    cat_cols = ["レース場", "風向", "天候"]
    for col in expected_features:
        if col in cat_cols and col in X_input.columns:
            saved_cats = cat_categories.get(col, None)
            if saved_cats:
                X_input[col] = pd.Categorical(X_input[col], categories=saved_cats)
            else:
                X_input[col] = X_input[col].astype("category")
        elif col not in cat_cols:
            X_input[col] = pd.to_numeric(X_input[col], errors="coerce")
            X_input[col] = X_input[col].fillna(
                feature_medians.get(col, 0.0)
                if isinstance(feature_medians, dict)
                else 0.0
            )

    prob_matrix = {}
    for rank_idx, rank_name in enumerate(["rank_1", "rank_2", "rank_3"], 1):
        if rank_name in models and models[rank_name] is not None:
            preds = models[rank_name].predict(X_input)
            if len(preds) > 0:
                prob_matrix[rank_idx] = np.array(preds[0])

    if not (1 in prob_matrix and 2 in prob_matrix and 3 in prob_matrix):
        return None

    m1, m2, m3 = prob_matrix[1], prob_matrix[2], prob_matrix[3]
    entry_courses = {i + 1: i + 1 for i in range(6)}
    default_kimarite_map = {
        1: "逃げ",
        2: "差し",
        3: "まくり",
        4: "まくり",
        5: "まくり差し",
        6: "まくり差し",
    }

    trifecta_scores = []
    total_score_sum = 0.0

    for c1_idx, c2_idx, c3_idx in itertools.permutations(range(6), 3):
        b1, b2, b3 = c1_idx + 1, c2_idx + 1, c3_idx + 1
        p1, p2, p3 = float(m1[c1_idx]), float(m2[c2_idx]), float(m3[c3_idx])
        ai_base_score = (p1**1.8) * (p2**1.3) * (p3**1.0)
        c1_course, c2_course, c3_course = (
            entry_courses[b1],
            entry_courses[b2],
            entry_courses[b3],
        )
        primary_kimarite = default_kimarite_map.get(b1, "差し")
        k_key = f"{primary_kimarite}_{c1_course}"
        pair_prob = kimarite_prob_dict.get(
            (k_key, c2_course, c3_course),
            kimarite_prob_dict.get((primary_kimarite, c2_course, c3_course), 0.001),
        )
        final_score = ai_base_score * (max(pair_prob, 0.001) ** 0.3)
        trifecta_scores.append(((b1, b2, b3), final_score))
        total_score_sum += final_score

    trifecta_scores.sort(key=lambda x: x[1], reverse=True)

    # -------------------------------------------------------------
    # 🎯 直前オッズ (od3) 解析 ＆ 期待値フィルター
    # -------------------------------------------------------------
    odds_map = {}
    if df_odds is not None and not df_odds.empty:
        for col in df_odds.columns:
            if "レースコード" in col or "code" in col.lower():
                col_vals = (
                    df_odds[col]
                    .astype(str)
                    .str.replace(r"\.0$", "", regex=True)
                    .str.strip()
                )
                matched_odds = df_odds[col_vals == target_race_code]
                if len(matched_odds) > 0:
                    matched_row = matched_odds.iloc[0]
                    for c1, c2, c3 in itertools.permutations(range(1, 7), 3):
                        combo_key = f"{c1}-{c2}-{c3}"
                        col_name = f"3連単_{combo_key}"
                        if col_name in matched_row and pd.notna(
                            matched_row[col_name]
                        ):
                            try:
                                val = float(
                                    str(matched_row[col_name])
                                    .replace(",", "")
                                    .strip()
                                )
                                if val > 0:
                                    odds_map[combo_key] = val
                            except ValueError:
                                pass
                    break

    # AI上位8点以内の中から期待値が高い買い目を抽出
    candidate_combos = trifecta_scores[:8]
    selected_combos = []

    MIN_PROBABILITY = 0.035  # AI推定確率 3.5% 以上
    MIN_EXPECTED_VALUE = 1.05  # 期待値 1.05 以上

    for combo, score in candidate_combos:
        combo_str = f"{combo[0]}-{combo[1]}-{combo[2]}"
        est_prob = score / total_score_sum if total_score_sum > 0 else 0.0

        # AI勝率が最低ライン以下の大穴は完全無視
        if est_prob < MIN_PROBABILITY:
            continue

        if combo_str in odds_map:
            odds = odds_map[combo_str]
            expected_value = est_prob * odds

            # 期待値条件クリア かつ ガミ回避（オッズ3.0倍以上）
            if expected_value >= MIN_EXPECTED_VALUE and odds >= 3.0:
                selected_combos.append(combo)

    # 採用する買い目（最大5点）
    final_combos = selected_combos[:5]

    top_score = trifecta_scores[0][1]

    # ステータス判定
    if len(final_combos) == 0:
        status = "見"
    elif top_score >= 0.0030:
        status = "勝負"
    else:
        status = "通常"

    return {
        "target_race_code": target_race_code,
        "status": status,
        "top5_combos": final_combos,
    }


def run_backtest(start_date_str, end_date_str, bet_per_combo=100):
    start_dt = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date_str, "%Y-%m-%d")

    stats = {
        "全": {"races": 0, "hits": 0, "invest": 0, "payout": 0},
        "勝負": {"races": 0, "hits": 0, "invest": 0, "payout": 0},
        "通常": {"races": 0, "hits": 0, "invest": 0, "payout": 0},
        "見": {"races": 0, "hits": 0, "invest": 0, "payout": 0},
    }

    logs = []
    curr_dt = start_dt

    print(f"🚀 バックテスト開始: {start_date_str} ～ {end_date_str}")

    while curr_dt <= end_dt:
        year = curr_dt.strftime("%Y")
        month_str = curr_dt.strftime("%m")
        day_str_zf = curr_dt.strftime("%d")
        day_str = curr_dt.strftime("%Y-%m-%d")
        month_raw = str(curr_dt.month)
        day_raw = str(curr_dt.day)

        res_p1 = f"data/results/payouts/{year}/{month_str}/{day_str_zf}.csv"
        res_p2 = f"data/results/payouts/{year}/{month_raw}/{day_raw}.csv"
        df_results = fetch_github_csv_with_fallback(res_p1, res_p2, use_cache=True)

        # 直前オッズ (od3) の取得
        odds_p1 = f"data/previews/od3/{year}/{month_str}/{day_str_zf}.csv"
        odds_p2 = f"data/previews/od3/{year}/{month_raw}/{day_raw}.csv"
        df_odds = fetch_github_csv_with_fallback(odds_p1, odds_p2, use_cache=True)

        for venue_name, venue_code in VENUE_MAPPING.items():
            for r_num in range(1, 13):
                pred = predict_single_race(
                    venue_code,
                    year,
                    month_str,
                    day_str_zf,
                    month_raw,
                    day_raw,
                    r_num,
                    df_odds=df_odds,
                )
                if not pred:
                    continue

                code = str(pred["target_race_code"]).strip()
                status = pred["status"]
                top5 = pred["top5_combos"]

                actual_combo = None
                payout = 0.0

                if df_results is not None and not df_results.empty:
                    matched = None
                    for col in df_results.columns:
                        if "レースコード" in col or "code" in col.lower():
                            col_vals = (
                                df_results[col]
                                .astype(str)
                                .str.replace(r"\.0$", "", regex=True)
                                .str.strip()
                            )
                            m = df_results[col_vals == code]
                            if len(m) > 0:
                                matched = m.iloc[0]
                                break

                    if matched is not None:
                        combo_val = str(matched.get("3連単_組番", "")).strip()
                        payout_val = matched.get("3連単_払戻金")

                        if combo_val and "-" in combo_val:
                            parts = combo_val.split("-")
                            if len(parts) == 3 and all(p.isdigit() for p in parts):
                                actual_combo = (
                                    int(parts[0]),
                                    int(parts[1]),
                                    int(parts[2]),
                                )

                        if pd.notna(payout_val):
                            try:
                                payout = float(
                                    str(payout_val)
                                    .replace(",", "")
                                    .replace("円", "")
                                    .strip()
                                )
                            except ValueError:
                                payout = 0.0

                is_hit = (actual_combo is not None) and (actual_combo in top5)

                if status == "見":
                    cost = 0
                    win_payout = 0.0
                    hit_label = "ー(見送り)"
                else:
                    cost = len(top5) * bet_per_combo
                    win_payout = payout if is_hit else 0.0
                    hit_label = "🎯的中" if is_hit else "❌不的中"

                if status != "見":
                    stats["全"]["races"] += 1
                    stats["全"]["invest"] += cost
                    stats["全"]["payout"] += win_payout
                    if is_hit:
                        stats["全"]["hits"] += 1

                if status in stats:
                    stats[status]["races"] += 1
                    if status != "見":
                        stats[status]["invest"] += cost
                        stats[status]["payout"] += win_payout
                        if is_hit:
                            stats[status]["hits"] += 1

                logs.append(
                    {
                        "日付": day_str,
                        "会場": venue_name,
                        "R": f"{r_num}R",
                        "ステータス": status,
                        "予想買い目": [f"{c[0]}-{c[1]}-{c[2]}" for c in top5],
                        "結果": (
                            f"{actual_combo[0]}-{actual_combo[1]}-{actual_combo[2]}"
                            if actual_combo
                            else "不明"
                        ),
                        "的中": hit_label,
                        "払戻金": win_payout,
                    }
                )

        curr_dt += timedelta(days=1)

    print("\n" + "=" * 50)
    print("📊 【バックテスト結果レポート】")
    print("=" * 50)

    def calc_rate(hits, races):
        return (hits / races * 100) if races > 0 else 0.0

    def calc_roi(payout, invest):
        return (payout / invest * 100) if invest > 0 else 0.0

    print(f"・購入対象レース数（勝負＋通常）: {stats['全']['races']} レース")
    print(f"・的中レース数: {stats['全']['hits']} レース")
    print(f"・購入対象的中率: {calc_rate(stats['全']['hits'], stats['全']['races']):.2f}%")
    print(
        f"・実質総投資: {stats['全']['invest']:,} 円 | 実質総払戻: {int(stats['全']['payout']):,} 円"
    )
    print(
        f"・実質回収率: {calc_roi(stats['全']['payout'], stats['全']['invest']):.2f}%\n"
    )

    print("--- 判定別レース数 ---")
    print(f"・🔥 勝負レース数: {stats['勝負']['races']} レース")
    print(f"・📊 通常レース数: {stats['通常']['races']} レース")
    print(f"・⚠️ 見（見送り）数: {stats['見']['races']} レース\n")

    print("--- 🔥 勝負レース単体の成績 ---")
    print(
        f"・的中率: {calc_rate(stats['勝負']['hits'], stats['勝負']['races']):.2f}%"
    )
    print(
        f"・回収率: {calc_roi(stats['勝負']['payout'], stats['勝負']['invest']):.2f}%\n"
    )

    df_log = pd.DataFrame(logs)
    if not df_log.empty:
        print(
            df_log[
                [
                    "日付",
                    "会場",
                    "R",
                    "ステータス",
                    "予想買い目",
                    "結果",
                    "的中",
                    "払戻金",
                ]
            ]
            .head(10)
            .to_string(index=False)
        )
        df_log.to_csv("backtest_results.csv", index=False, encoding="utf-8-sig")
        print("\n📁 詳細なバックテスト結果を 'backtest_results.csv' に出力しました。")


if __name__ == "__main__":
    run_backtest("2026-09-13", "2026-09-13")
