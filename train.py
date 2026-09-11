import os
import glob
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import joblib
from datetime import datetime

def load_csv_safely(path):
    if os.path.exists(path):
        try:
            df = pd.read_csv(path, dtype=str)
            df.columns = df.columns.str.strip()
            return df
        except Exception:
            pass
    return None

def get_target_files(result_files):
    """
    2026年3月1日から現在までのデータを自動で取得・更新する
    """
    start_date = datetime(2026, 3, 1)
    current_date = datetime.now()  # 実行時の現在時刻を上限として自動反映
    
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
        print("パスからの日付抽出ができなかったため、すべてのファイルを採用します。")
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
        print("警告: resultsデータから「決まり手」カラムが見つかりませんでした。")
        return {}

    player_col = next((col for col in df_all_res.columns if "1着_選手名" in col or ("選手名" in col and "1着" in col)), None)
    if not player_col:
        player_col = next((col for col in df_all_res.columns if "選手名" in col), None)

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

    merged_rows = []
    file_cache = {}

    for res_file in target_files:
        try:
            df_res = pd.read_csv(res_file, dtype=str)
            df_res.columns = df_res.columns.str.strip()
            
            for _, res_row in df_res.iterrows():
                r_code = str(res_row.get("レースコード", "")).strip()
                if len(r_code) < 12:
                    continue
                
                year = r_code[0:4]
                month = r_code[4:6]
                day = r_code[6:8]

                race_card_path = f"data/programs/race_cards/{year}/{month}/{day}.csv"
                sui_path = f"data/previews/sui/{year}/{month}/{day}.csv"
                orig_path = f"data/previews/original_exhibition/{year}/{month}/{day}.csv"

                if race_card_path not in file_cache:
                    file_cache[race_card_path] = load_csv_safely(race_card_path)
                    file_cache[sui_path] = load_csv_safely(sui_path)
                    file_cache[orig_path] = load_csv_safely(orig_path)

                df_cards = file_cache.get(race_card_path)
                df_sui = file_cache.get(sui_path)
                df_orig = file_cache.get(orig_path)

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

                combined_row = {}
                combined_row.update(card_row)
                combined_row.update(sui_row)
                combined_row.update(orig_row)
                
                for k, v in res_row.items():
                    combined_row[f"res_{k}"] = v

                merged_rows.append(combined_row)

        except Exception as e:
            print(f"ファイル処理エラー ({res_file}): {e}")

    if not merged_rows:
        return None

    return pd.DataFrame(merged_rows)

def train_model():
    player_fav_kimarite = build_player_kimarite_stats()
    df_train = load_and_merge_training_data()

    if df_train is None or len(df_train) == 0:
        print("有効な学習データがありません。処理を中断します。")
        return

    if player_fav_kimarite:
        for i in range(1, 7):
            p_col_candidates = [f"艇{i}_選手名", f"{i}号艇_選手名", f"選手名_{i}", f"選手{i}_名前"]
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

    targets = ["res_1着_艇番", "res_2着_艇番", "res_3着_艇番"]
    for t in targets:
        if t in df_train.columns:
            df_train[t] = pd.to_numeric(df_train[t], errors='coerce')

    df_train = df_train.dropna(subset=[t for t in targets if t in df_train.columns])

    for col in feature_cols:
        if col not in ["レース場", "風向", "天候"] and pd.api.types.is_numeric_dtype(df_train[col]):
            df_train[col] = df_train[col].fillna(df_train[col].median())

    print(f"有効な学習レース数: {len(df_train)}行")

    if len(df_train) == 0:
        print("エラー: 有効なデータ行が0件です。")
        return

    features = [col for col in feature_cols if col not in ["レースコード", "選手名"]]
    X = df_train[features]
    models = {}

    for i, target_col in enumerate(["res_1着_艇番", "res_2着_艇番", "res_3着_艇番"], start=1):
        if target_col not in df_train.columns:
            continue
            
        print(f"--- {i}着の予測モデルを学習中 ({target_col}) ---")
        y = df_train[target_col].astype(int) - 1

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

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

    print("\n--- モデルの検証結果（正解率の確認） ---")
    for i, target_col in enumerate(["res_1着_艇番", "res_2着_艇番", "res_3着_艇番"], start=1):
        if f"rank_{i}" in models and target_col in df_train.columns:
            _, X_val, _, y_val = train_test_split(X, df_train[target_col].astype(int) - 1, test_size=0.2, random_state=42)
            preds = models[f"rank_{i}"].predict(X_val)
            pred_labels = np.argmax(preds, axis=1)
            accuracy = np.mean(pred_labels == y_val) * 100
            print(f"🔹 {i}着予想の正解率: {accuracy:.2f}%")

    saved_package = {
        "models": models,
        "player_course_stats": None,
        "venue_wind_kimarite": None,
        "player_fav_kimarite": player_fav_kimarite,
        "player_col": "選手名"
    }

    joblib.dump(saved_package, "boatrace_lgb_model.pkl")
    print("モデルと決まり手集計データを 'boatrace_lgb_model.pkl' に保存しました。")

if __name__ == "__main__":
    train_model()

