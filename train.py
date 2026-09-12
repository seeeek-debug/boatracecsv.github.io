import os
import glob
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
import joblib
from datetime import datetime
import itertools

def load_csv_safely(path):
    if os.path.exists(path):
        try:
            df = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
            df.columns = df.columns.str.strip()
            return df
        except Exception:
            pass
    return None

def get_target_files(result_files):
    start_date = datetime(2026, 3, 1)
    current_date = datetime.now()
    
    target_files = []
    for f in result_files:
        try:
            parts = f.replace("\\", "/").split("/")
            file_year, file_month, file_day = None, None, None
            
            for part in parts:
                if len(part) == 4 and part.isdigit() and 2020 <= int(part) <= 2030:
                    file_year = int(part)
                elif len(part) == 2 and part.isdigit() and 1 <= int(part) <= 12:
                    if file_year and not file_month:
                        file_month = int(part)
                    elif file_month and not file_day:
                        file_day = int(part)
            
            filename = os.path.splitext(os.path.basename(f))[0]
            if len(filename) == 2 and filename.isdigit() and file_year and file_month:
                file_day = int(filename)

            if file_year and file_month:
                file_date = datetime(file_year, file_month, file_day if file_day else 1)
                if start_date <= file_date <= current_date:
                    target_files.append(f)
                    continue
        except Exception:
            pass
            
    if not target_files:
        target_files = result_files
        
    print(f"2026年3月1日以降の対象ファイル数: {len(target_files)}件")
    return target_files

def build_player_kimarite_stats():
    print("選手ごとの得意な決まり手の集計を開始します...")
    result_files = glob.glob("data/results/**/*.csv", recursive=True)
    if not result_files:
        result_files = glob.glob("data/**/*.csv", recursive=True)

    target_files = get_target_files(result_files)

    dfs = []
    for f in target_files:
        df = load_csv_safely(f)
        if df is not None:
            dfs.append(df)
            
    if not dfs:
        return {}

    df_all_res = pd.concat(dfs, ignore_index=True)
    
    kimarite_col = next((col for col in df_all_res.columns if "決まり手" in col), None)
    if not kimarite_col:
        return {}

    player_col_candidates = ["1着_選手名", "選手名", "氏名", "1着_氏名"]
    player_col = next((col for col in df_all_res.columns if any(c in col for c in player_col_candidates)), None)

    player_stats = {}
    if player_col and kimarite_col:
        grouped = df_all_res.groupby([player_col, kimarite_col]).size().unstack(fill_value=0)
        grouped_rate = grouped.div(grouped.sum(axis=1), axis=0)
        player_stats = grouped_rate.to_dict(orient="index")
        print(f"選手ごとの決まり手データを集計しました（対象選手数: {len(player_stats)}人）")

    return player_stats

def load_and_merge_training_data():
    print("横持ちファイルの読み込みと結合を開始します...")
    
    result_files = glob.glob("data/results/**/*.csv", recursive=True)
    if not result_files:
        result_files = glob.glob("data/**/*.csv", recursive=True)

    target_files = get_target_files(result_files)

    df_course_win = load_csv_safely("data/estimate/stadium/course_win_rate.csv")
    df_season_win = load_csv_safely("data/estimate/stadium/win_rate.csv")
    
    course_win_dict = {}
    if df_course_win is not None:
        for _, row in df_course_win.iterrows():
            v_code = str(row.get("場コード", "")).strip().zfill(2)
            r_num = str(row.get("レース回", "")).strip()
            course_win_dict[(v_code, r_num)] = row.to_dict()

    season_win_dict = {}
    if df_season_win is not None:
        for _, row in df_season_win.iterrows():
            v_code = str(row.get("場コード", "")).strip().zfill(2)
            season = str(row.get("季節", "")).strip()
            season_win_dict[(v_code, season)] = row.to_dict()

    venue_preview_code_map = {
        "01": "kir", "02": "tod", "03": "edg", "04": "hei", "05": "tam", "06": "ham",
        "07": "gam", "08": "tkz", "09": "tsu", "10": "mik", "11": "biw", "12": "sum",
        "13": "ama", "14": "nar", "15": "mar", "16": "koj", "17": "miy", "18": "tok",
        "19": "shm", "20": "wkm", "21": "ash", "22": "fuk", "23": "ktu", "24": "omr"
    }

    merged_rows = []
    file_cache = {}

    def get_season(m):
        if m in [3, 4, 5]: return "春"
        elif m in [6, 7, 8]: return "夏"
        elif m in [9, 10, 11]: return "秋"
        else: return "冬"

    for res_file in target_files:
        try:
            df_res = pd.read_csv(res_file, dtype=str, encoding="utf-8-sig")
            df_res.columns = df_res.columns.str.strip()
            
            for _, res_row in df_res.iterrows():
                r_code = str(res_row.get("レースコード", "")).strip()
                if len(r_code) < 12:
                    continue
                
                year = r_code[0:4]
                month = int(r_code[4:6])
                day = r_code[6:8]
                venue_code = r_code[8:10]
                race_round = str(int(r_code[10:12]))

                race_card_path = f"data/programs/race_cards/{year}/{f'{month:02d}'}/{day}.csv"
                sui_path = f"data/previews/sui/{year}/{f'{month:02d}'}/{day}.csv"
                orig_path = f"data/previews/original_exhibition/{year}/{f'{month:02d}'}/{day}.csv"
                stt_path = f"data/previews/stt/{year}/{f'{month:02d}'}/{day}.csv"
                
                prev_code = venue_preview_code_map.get(venue_code, "")
                venue_preview_path = f"data/previews/{prev_code}/{year}/{f'{month:02d}'}/{day}.csv" if prev_code else ""

                if race_card_path not in file_cache:
                    file_cache[race_card_path] = load_csv_safely(race_card_path)
                    file_cache[sui_path] = load_csv_safely(sui_path)
                    file_cache[orig_path] = load_csv_safely(orig_path)
                    file_cache[stt_path] = load_csv_safely(stt_path)
                    if venue_preview_path and venue_preview_path not in file_cache:
                        file_cache[venue_preview_path] = load_csv_safely(venue_preview_path)

                df_cards = file_cache.get(race_card_path)
                df_sui = file_cache.get(sui_path)
                df_orig = file_cache.get(orig_path)
                df_stt = file_cache.get(stt_path)
                df_venue_preview = file_cache.get(venue_preview_path) if venue_preview_path else None

                if df_cards is None:
                    continue

                def get_matched_row(df, code):
                    if df is None: return None
                    for col in df.columns:
                        if "レースコード" in col or "code" in col.lower():
                            matched = df[df[col].astype(str).str.strip() == str(code)]
                            if len(matched) > 0:
                                return matched.iloc[0].to_dict()
                    return None

                card_row = get_matched_row(df_cards, r_code)
                if not card_row:
                    continue

                sui_row = get_matched_row(df_sui, r_code) or {}
                orig_row = get_matched_row(df_orig, r_code) or {}
                stt_row = get_matched_row(df_stt, r_code) or {}
                venue_preview_row = get_matched_row(df_venue_preview, r_code) or {}

                combined_row = {}
                combined_row.update(card_row)
                combined_row.update(sui_row)
                combined_row.update(orig_row)
                combined_row.update(stt_row)
                combined_row.update(venue_preview_row)
                
                c_data = course_win_dict.get((venue_code, race_round), {})
                for k, v in c_data.items():
                    if k not in ["場コード", "レース回"]:
                        combined_row[f"est_course_{k}"] = v

                season_name = get_season(month)
                s_data = season_win_dict.get((venue_code, season_name), {})
                for k, v in s_data.items():
                    if k not in ["場コード", "季節"]:
                        combined_row[f"est_season_{k}"] = v

                for k, v in res_row.items():
                    combined_row[f"res_{k}"] = v

                merged_rows.append(combined_row)

        except Exception as e:
            pass

    if not merged_rows:
        return None

    return pd.DataFrame(merged_rows)

def train_model():
    player_fav_kimarite = build_player_kimarite_stats()
    df_train = load_and_merge_training_data()

    if df_train is None or len(df_train) == 0:
        print("有効な学習データがありません。")
        return

    # 表記揺れに対応して決まり手特徴量を算出
    if player_fav_kimarite:
        for i in range(1, 7):
            p_col_candidates = [
                f"艇{i}_選手名", f"{i}号艇_選手名", f"選手名_{i}", f"選手{i}_名前",
                f"艇{i}_氏名", f"{i}号艇_氏名", f"氏名_{i}", f"艇{i}_選手", f"{i}号艇_選手"
            ]
            p_col = next((c for c in p_col_candidates if c in df_train.columns), None)
            
            if p_col:
                dummy_k_keys = list(next(iter(player_fav_kimarite.values())).keys()) if player_fav_kimarite else []
                for k_name in dummy_k_keys:
                    col_name = f"艇{i}_kimarite_{k_name}"
                    df_train[col_name] = df_train[p_col].map(
                        lambda name: player_fav_kimarite.get(str(name).strip(), {}).get(k_name, 0.0)
                    )

    exclude_cols = [col for col in df_train.columns if col.startswith("res_")]
    feature_cols = [col for col in df_train.columns if col not in exclude_cols]

    for col in feature_cols:
        if col not in ["レース場", "風向", "天候"]:
            df_train[col] = pd.to_numeric(df_train[col], errors='coerce')

    for col in ["レース場", "風向", "天候"]:
        if col in df_train.columns:
            df_train[col] = df_train[col].astype('category')

    targets = ["res_1着_艇番", "res_2着_艇番", "res_3着_艇番"]
    for t in targets:
        if t in df_train.columns:
            df_train[t] = pd.to_numeric(df_train[t], errors='coerce')

    df_train = df_train.dropna(subset=[t for t in targets if t in df_train.columns])

    # 学習データの各数値カラム中央値を計算・保存用辞書へ格納（推論側補完用）
    feature_medians = {}
    for col in feature_cols:
        if col not in ["レース場", "風向", "天候"] and pd.api.types.is_numeric_dtype(df_train[col]):
            median_val = df_train[col].median()
            feature_medians[col] = median_val
            df_train[col] = df_train[col].fillna(median_val)

    print(f"有効な学習レース数: {len(df_train)}行")
    if len(df_train) == 0:
        return

    features = [col for col in feature_cols if col not in ["レースコード", "選手名"]]
    X = df_train[features]

    targets_df = df_train[["res_1着_艇番", "res_2着_艇番", "res_3着_艇番"]].astype(int)
    X_train, X_val, y_train_df, y_val_df = train_test_split(X, targets_df, test_size=0.2, random_state=42)

    models = {}
    for i, target_col in enumerate(["res_1着_艇番", "res_2着_艇番", "res_3着_艇番"], start=1):
        print(f"--- {i}着の予測モデルを学習中 ({target_col}) ---")
        y_train = y_train_df[target_col] - 1
        y_val = y_val_df[target_col] - 1

        train_weights = compute_sample_weight('balanced', y_train)
        val_weights = compute_sample_weight('balanced', y_val)

        train_data = lgb.Dataset(X_train, label=y_train, weight=train_weights)
        val_data = lgb.Dataset(X_val, label=y_val, weight=val_weights, reference=train_data)

        params = {
            "objective": "multiclass",
            "num_class": 6,
            "metric": "multi_logloss",
            "boosting_type": "gbdt",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "random_state": 42
        }

        model = lgb.train(
            params,
            train_data,
            num_boost_round=200,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
        )
        models[f"rank_{i}"] = model

    # キー名を推論ボット側と完全同一にして保存
    saved_package = {
        "models": models,
        "feature_names": features,
        "feature_medians": feature_medians,
        "player_fav_kimarite": player_fav_kimarite,
        "player_col": "選手名"
    }

    joblib.dump(saved_package, "boatrace_lgb_model.pkl")
    print("モデル、特徴量名、中央値データを 'boatrace_lgb_model.pkl' に正常保存しました。")

if __name__ == "__main__":
    train_model()

