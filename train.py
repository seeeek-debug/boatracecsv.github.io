import os
import glob
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import joblib

def load_and_merge_data():
    print("CSVデータの読み込みを開始します（2026年3月以降のデータ）...")
    
    result_files = glob.glob("data/results/**/*.csv", recursive=True)
    if not result_files:
        result_files = glob.glob("data/**/*.csv", recursive=True)
        
    if not result_files:
        print("エラー: データファイルが見つかりません。")
        return None

    target_files = []
    for file in result_files:
        if "2026/01" in file or "2026/02" in file or "2026-01" in file or "2026-02" in file:
            continue
        if "2025" in file or "2024" in file:
            continue
            
        if "2026/" in file or "2026-" in file or any(f"202{y}/" in file for y in range(7, 10)):
            target_files.append(file)

    print(f"2026年3月以降の対象ファイル数: {len(target_files)}件")
    
    if not target_files:
        print("警告: 条件に一致するファイルが見つかりなかったため、最新の50件を使用します。")
        target_files = sorted(result_files)[-50:]

    df_list = []
    for file in target_files:
        try:
            df = pd.read_csv(file)
            df.columns = df.columns.str.strip()
            df_list.append(df)
        except Exception as e:
            print(f"ファイル読み込みスキップ ({file}): {e}")
            
    if not df_list:
        return None
        
    print("データを結合しています...")
    df_base = pd.concat(df_list, ignore_index=True)
    
    if len(df_base) > 100000:
        print(f"データ数が多いため（{len(df_base)}行）、10万行にサンプリングします。")
        df_base = df_base.sample(n=100000, random_state=42)
        
    return df_base

def train_model():
    df_train = load_and_merge_data()
    
    if df_train is None or len(df_train) == 0:
        print("有効な学習データがありません。処理を中断します。")
        return

    target_features = [
        "レース場",
        "風速(m)",
        "波の高さ(cm)",
        "水温(°C)",
        "気温(°C)",
        "風向",
        "天候",
        "1コース_スタートタイミング",
        "2コース_スタートタイミング",
        "3コース_スタートタイミング",
        "4コース_スタートタイミング",
        "5コース_スタートタイミング",
        "6コース_スタートタイミング",
        "回り足",
        "直線",
        "一周タイム",
        "半周タイム",
    ]
    
    features = [col for col in target_features if col in df_train.columns]
    print(f"実際に使用する特徴量: {features}")

    target_candidates = [
        "1着_艇番", "2着_艇番", "3着_艇番", 
        "4着_艇番", "5着_艇番", "6着_艇番"
    ]
    targets = [col for col in target_candidates if col in df_train.columns]
    
    if not targets:
        print("エラー: 目的変数（着番データ）が見つかりません。")
        return

    # 1. まず特徴量の各列を強制的に数値型に変換（Fや文字などはNaNになる）
    for col in features:
        df_train[col] = pd.to_numeric(df_train[col], errors='coerce')

    # 2. 欠損値（NaNになったものや元々ないもの）をまとめて除外
    df_train = df_train.dropna(subset=targets + features)
    print(f"欠損値・文字混入データ除外後の有効データ数: {len(df_train)}行")
    
    if len(df_train) == 0:
        print("エラー: 有効なデータ行が0件です。")
        return

    X = df_train[features]
    models = {}

    for i, target_col in enumerate(targets, start=1):
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
            "random_state": 42,
            "n_jobs": -1
        }

        model = lgb.train(
            params,
            train_data,
            num_boost_round=150,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(20)]
        )
        
        models[f"rank_{i}"] = model

    model_filename = "boatrace_lgb_model.pkl"
    joblib.dump(models, model_filename)
    print(f"学習完了！モデルを {model_filename} として保存しました。")

if __name__ == "__main__":
    train_model()

