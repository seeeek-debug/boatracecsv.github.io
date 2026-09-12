import os
import joblib
import pandas as pd
import numpy as np
import itertools

# --- 基本設定 ---
MODEL_FILENAME = "boatrace_lgb_model.pkl"
TARGET_DATE = "2026-09-11"

VENUES = [
    "桐生", "戸田", "江戸川", "平和島", "多摩川", "浜名湖",
    "蒲郡", "常滑", "津", "三国", "びわこ", "住之江",
    "尼崎", "鳴門", "丸亀", "児島", "宮島", "徳山",
    "下関", "若松", "芦屋", "福岡", "唐津", "大村"
]

VENUE_MAPPING = {
    "桐生": "01", "戸田": "02", "江戸川": "03", "平和島": "04", "多摩川": "05", "浜名湖": "06",
    "蒲郡": "07", "常滑": "08", "津": "09", "三国": "10", "びわこ": "11", "住之江": "12",
    "尼崎": "13", "鳴門": "14", "丸亀": "15", "児島": "16", "宮島": "17", "徳山": "18",
    "下関": "19", "若松": "20", "芦屋": "21", "福岡": "22", "唐津": "23", "大村": "24"
}

VENUE_PREVIEW_CODE_MAP = {
    "01": "kir", "02": "tod", "03": "edg", "04": "hei", "05": "tam", "06": "ham",
    "07": "gam", "08": "tkz", "09": "tsu", "10": "mik", "11": "biw", "12": "sum",
    "13": "ama", "14": "nar", "15": "mar", "16": "koj", "17": "miy", "18": "tok",
    "19": "shm", "20": "wkm", "21": "ash", "22": "fuk", "23": "ktu", "24": "omr"
}

CSV_CACHE = {}

def load_csv(file_path):
    if not file_path:
        return None
    if file_path in CSV_CACHE:
        return CSV_CACHE[file_path]
    
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path, encoding="utf-8-sig", dtype=str)
            df.columns = df.columns.str.strip()
            CSV_CACHE[file_path] = df
            return df
        except Exception:
            pass
    return None

def get_first_available(paths):
    for p in paths:
        if not p:
            continue
        df = load_csv(p)
        if df is not None:
            return df, p
    return None, None

def get_season(m):
    if m in [3, 4, 5]: return "春"
    elif m in [6, 7, 8]: return "夏"
    elif m in [9, 10, 11]: return "秋"
    else: return "冬"

# --- 1. モデルおよび決まり手データのロード ---
print("📦 ローカルからモデルおよび環境データを読み込んでいます...")
if not os.path.exists(MODEL_FILENAME):
    print(f"❌ エラー: モデルファイル '{MODEL_FILENAME}' が見つかりません。")
    exit(1)

loaded_package = joblib.load(MODEL_FILENAME)
models = {}
expected_features = []
player_fav_kimarite = None
kimarite_prob_dict = {}

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
    elif "rank_1" in models and hasattr(models["rank_1"], "feature_name"):
        expected_features = models["rank_1"].feature_name()

if not kimarite_prob_dict:
    df_pair = load_csv("data/estimate/kimarite/tables/pair_table.csv")
    if df_pair is not None:
        for _, row in df_pair.iterrows():
            k_type = str(row['セル']).strip()
            c2 = int(row['2着コース'])
            c3 = int(row['3着コース'])
            kimarite_prob_dict[(k_type, c2, c3)] = float(row['確率'])

# --- 2. バックテスト実行 ---
year, month, day = TARGET_DATE.split("-")
month_str, day_str = month.zfill(2), day.zfill(2)
month_raw, day_raw = str(int(month)), str(int(day))

print(f"\n🚀 {TARGET_DATE} の全レース検証を開始します...\n")

total_races = 0
hits = 0
results_summary = []
found_any_card = False

for venue in VENUES:
    venue_s = VENUE_MAPPING[venue]
    prev_code = VENUE_PREVIEW_CODE_MAP.get(venue_s, "")

    card_paths = [
        f"data/programs/race_cards/{year}/{month_str}/{day_str}.csv",
        f"data/programs/race_cards/{year}/{month_raw}/{day_raw}.csv",
        f"data/race_cards/{year}/{month_str}/{day_str}.csv",
        f"data/race_cards/{year}/{month_raw}/{day_raw}.csv"
    ]
    # payouts 階層を含むファイルパスを追加
    result_paths = [
        f"data/results/payouts/{year}/{month_str}/{day_str}.csv",
        f"data/results/payouts/{year}/{month_raw}/{day_raw}.csv",
        f"data/results/{year}/{month_str}/{day_str}.csv",
        f"data/results/{year}/{month_raw}/{day_raw}.csv",
        f"data/results/{year}/{month_str}/{month_str}{day_str}.csv"
    ]
    sui_paths = [
        f"data/previews/sui/{year}/{month_str}/{day_str}.csv",
        f"data/previews/sui/{year}/{month_raw}/{day_raw}.csv"
    ]
    orig_paths = [
        f"data/previews/original_exhibition/{year}/{month_str}/{day_str}.csv",
        f"data/previews/original_exhibition/{year}/{month_raw}/{day_raw}.csv"
    ]
    stt_paths = [
        f"data/previews/stt/{year}/{month_str}/{day_str}.csv",
        f"data/previews/stt/{year}/{month_raw}/{day_raw}.csv"
    ]
    venue_preview_paths = [
        f"data/previews/{prev_code}/{year}/{month_str}/{day_str}.csv" if prev_code else "",
        f"data/previews/{prev_code}/{year}/{month_raw}/{day_raw}.csv" if prev_code else ""
    ]

    df_cards, _ = get_first_available(card_paths)
    if df_cards is None:
        continue
    found_any_card = True

    df_results, _ = get_first_available(result_paths)
    df_sui, _ = get_first_available(sui_paths)
    df_orig, _ = get_first_available(orig_paths)
    df_stt, _ = get_first_available(stt_paths)
    df_venue_preview, _ = get_first_available(venue_preview_paths)

    if df_orig is not None:
        rename_dict = {f"艇{i}_値1": f"艇{i}_オリジナル一周タイム" for i in range(1, 7)}
        df_orig = df_orig.rename(columns=rename_dict)

    df_course_win = load_csv("data/estimate/stadium/course_win_rate.csv")
    df_season_win = load_csv("data/estimate/stadium/win_rate.csv")

    for r_num in range(1, 13):
        r_str = str(r_num).zfill(2)
        target_race_code = f"{year}{month_str}{day_str}{venue_s}{r_str}"

        def get_matched_row(df, code):
            if df is None: return None
            code_str = str(code).strip()
            for col in df.columns:
                if any(k in col for k in ["レースコード", "race_code", "コード", "code"]):
                    col_vals = df[col].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
                    matched = df[col_vals == code_str]
                    if len(matched) > 0:
                        return matched.iloc[0].to_dict()
            return None

        card_row = get_matched_row(df_cards, target_race_code)
        if not card_row:
            continue

        res_row = get_matched_row(df_results, target_race_code) or {}
        sui_row = get_matched_row(df_sui, target_race_code) or {}
        orig_row = get_matched_row(df_orig, target_race_code) or {}
        stt_row = get_matched_row(df_stt, target_race_code) or {}
        venue_preview_row = get_matched_row(df_venue_preview, target_race_code) or {}

        combined_row = {}
        combined_row.update(card_row)
        combined_row.update(res_row)
        combined_row.update(sui_row)
        combined_row.update(orig_row)
        combined_row.update(stt_row)
        combined_row.update(venue_preview_row)

        if df_course_win is not None:
            for _, row in df_course_win.iterrows():
                if str(row.get("場コード", "")).strip().zfill(2) == venue_s and str(row.get("レース回", "")).strip() == str(r_num):
                    for k, v in row.items():
                        if k not in ["場コード", "レース回"]: combined_row[f"est_course_{k}"] = v
                    break

        if df_season_win is not None:
            season_name = get_season(int(month))
            for _, row in df_season_win.iterrows():
                if str(row.get("場コード", "")).strip().zfill(2) == venue_s and str(row.get("季節", "")).strip() == season_name:
                    for k, v in row.items():
                        if k not in ["場コード", "季節"]: combined_row[f"est_season_{k}"] = v
                    break

        expanded_row = dict(combined_row)
        for k, v in list(combined_row.items()):
            for b in range(1, 7):
                sb = str(b)
                if k.startswith(f"艇{sb}_"):
                    expanded_row[f"{sb}号艇_{k[2:]}"] = v
                    expanded_row[f"{k[2:]}_{sb}"] = v
                elif k.startswith(f"{sb}号艇_"):
                    expanded_row[f"艇{sb}_{k[3:]}"] = v
                    expanded_row[f"{k[3:]}_{sb}"] = v

        df_pred = pd.DataFrame([expanded_row])

        if player_fav_kimarite:
            for i in range(1, 7):
                p_col = next((c for c in [f"艇{i}_選手名", f"{i}号艇_選手名", f"選手名_{i}"] if c in df_pred.columns), None)
                if p_col:
                    dummy_k_keys = list(next(iter(player_fav_kimarite.values())).keys()) if player_fav_kimarite else []
                    for k_name in dummy_k_keys:
                        p_val = str(df_pred.iloc[0].get(p_col, "")).strip()
                        df_pred[f"艇{i}_kimarite_{k_name}"] = player_fav_kimarite.get(p_val, {}).get(k_name, 0.0)

        for col in [c for c in df_pred.columns if not c.startswith("res_")]:
            if col not in ["レース場", "風向", "天候"]:
                df_pred[col] = pd.to_numeric(df_pred[col], errors='coerce')

        for col in ["レース場", "風向", "天候"]:
            if col in df_pred.columns:
                df_pred[col] = df_pred[col].astype('category')

        X_input = df_pred.reindex(columns=expected_features, fill_value=0.0)
        for col in expected_features:
            if col in ["レース場", "風向", "天候"] and col in X_input.columns:
                X_input[col] = X_input[col].astype('category')
        for col in X_input.select_dtypes(include=[np.number]).columns:
            X_input[col] = X_input[col].fillna(0.0)

        prob_matrix = {}
        for rank_idx, rank_name in enumerate(["rank_1", "rank_2", "rank_3"], 1):
            if rank_name in models and models[rank_name] is not None:
                preds = models[rank_name].predict(X_input)
                if len(preds) > 0:
                    prob_matrix[rank_idx] = np.array(preds[0])

        m1, m2, m3 = prob_matrix.get(1), prob_matrix.get(2), prob_matrix.get(3)
        if m1 is None or m2 is None or m3 is None:
            continue

        default_kimarite_map = {1: "逃げ", 2: "差し", 3: "まくり", 4: "まくり", 5: "まくり差し", 6: "差し"}
        trifecta_scores = []
        for c1_idx, c2_idx, c3_idx in itertools.permutations(range(6), 3):
            b1, b2, b3 = c1_idx + 1, c2_idx + 1, c3_idx + 1
            ai_base_score = (float(m1[c1_idx]) ** 1.8) * (float(m2[c2_idx]) ** 1.3) * (float(m3[c3_idx]) ** 1.0)
            primary_kimarite = default_kimarite_map.get(b1, "差し")
            pair_prob = kimarite_prob_dict.get((f"{primary_kimarite}_{b1}", b2, b3),
                        kimarite_prob_dict.get((primary_kimarite, b2, b3), 0.01))
            final_score = ai_base_score * (max(pair_prob, 0.001) ** 0.3)
            trifecta_scores.append((f"{b1}-{b2}-{b3}", final_score))

        trifecta_scores.sort(key=lambda x: x[1], reverse=True)
        top5_bets = [x[0] for x in trifecta_scores[:5]]

        # 実際の確定結果・払戻金の取得（新しいCSVの列名 3連単_組番 に対応）
        r1 = str(expanded_row.get("res_1着", "")).replace(".0", "").strip()
        r2 = str(expanded_row.get("res_2着", "")).replace(".0", "").strip()
        r3 = str(expanded_row.get("res_3着", "")).replace(".0", "").strip()
        
        actual_result = ""
        if r1 and r2 and r3 and r1 != "nan" and r2 != "nan" and r3 != "nan":
            actual_result = f"{r1}-{r2}-{r3}"
        else:
            actual_result = str(expanded_row.get("3連単_組番", expanded_row.get("res_3連単", ""))).replace(".0", "").strip()

        if not actual_result or actual_result in ["nan--", "nan"]:
            continue

        total_races += 1
        is_hit = actual_result in top5_bets
        if is_hit:
            hits += 1
            hit_rank = top5_bets.index(actual_result) + 1
            rank_str = f"🎯 的中 ({hit_rank}番手)"
        else:
            rank_str = "❌ 不的中"

        race_label = f"{venue} {r_num}R"
        payout = str(expanded_row.get("3連単_払戻金", expanded_row.get("res_3連単払戻", expanded_row.get("res_払戻", "-")))).strip()

        results_summary.append({
            "race": race_label,
            "bets": ", ".join(top5_bets),
            "actual": actual_result,
            "status": rank_str,
            "payout": payout
        })

# --- 3. 結果の表示 ---
print("==========================================================================")
print(f" 📊 {TARGET_DATE} 全レース AI予想バックテスト結果")
print("==========================================================================")

if not found_any_card:
    print(f"⚠️ 指定日 ({TARGET_DATE}) の出走表CSVが見つかりませんでした。")
elif total_races == 0:
    print(f"⚠️ レース結果データが見つかりませんでした。")
else:
    for r in results_summary:
        payout_info = f" | 払戻: {r['payout']}円" if "🎯" in r['status'] and r['payout'] != "-" else ""
        print(f"[{r['race']}] 予想5点: [{r['bets']}] | 確定: {r['actual']} | {r['status']}{payout_info}")

    print("--------------------------------------------------------------------------")
    hit_rate = (hits / total_races * 100) if total_races > 0 else 0
    print(f"総検証レース数 : {total_races} R")
    print(f"的中数         : {hits} R")
    print(f"的中率         : {hit_rate:.1f} %")
    print("==========================================================================")

