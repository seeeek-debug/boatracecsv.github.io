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
        if "2025" in file or "2024" in file:
            continue
        if "2026" in file:
            target_files.append(file)

    print(f"対象ファイル数: {len(target_files)}")

    if not target_files:
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
    return df_base

def train_model():
    df_train = load_and_merge_data()

    if df_train is None or len(df_train) == 0:
        print("有効な学習データがありません。処理を中断します。")
        return

    # 決まり手の集計
    kimarite_col = None
    for col in ["決まり手", "決まり手 (逃げ・まくり等)"]:
        if col in df_train.columns:
            kimarite_col = col
            break

    venue_wind_kimarite = None
    if kimarite_col:
        print("決まり手をエンコード・集計しています...")
        df_train["決まり手_コード"] = df_train[kimarite_col].astype('category')

        if "レース場" in df_train.columns and "風向" in df_train.columns:
            venue_wind_kimarite = df_train.groupby(["レース場", "風向"]).size().reset_index(name="場_風別_決まり手確率")
            df_train = pd.merge(df_train, venue_wind_kimarite, on=["レース場", "風向"], how="left")

    # 2026年リアルタイムCSV（横持ち）に存在する特徴量
    target_features = [
        "レース場",
        "風速(m)",
        "波の高さ(cm)",
        "水温(℃)",
        "気温(℃)",
        "風向",
        "天候",
        "1コース_スタートタイミング",
        "2コース_スタートタイミング",
        "3コース_スタートタイミング",
        "4コース_スタートタイミング",
        "5コース_スタートタイミング",
        "6コース_スタートタイミング",
        "場_風別_決まり手確率",
        "決まり手_コード"
    ]

    features = [col for col in target_features if col in df_train.columns]
    print(f"実際に使用する特徴量: {features}")

    target_candidates = [
        "1着_艇番", "2着_艇番", "3着_艇番",
        "4着_艇番", "5着_艇番", "6着_艇番"
    ]
    targets = [col for col in target_candidates if col in df_train.columns]

    # 数値変換
    for col in features:
        if col not in ["決まり手_コード", "レース場", "風向", "天候"]:
            df_train[col] = pd.to_numeric(df_train[col], errors='coerce')

    for col in targets:
        df_train[col] = pd.to_numeric(df_train[col], errors='coerce')

    # ターゲット（1〜3着）が確実に存在するものだけに絞る（特徴量は多少の欠損を許容）
    df_train = df_train.dropna(subset=targets)
    
    # 特徴量の欠損は中央値などで穴埋めしてデータが0行になるのを防ぐ
    for col in features:
        if col not in ["決まり手_コード", "レース場", "風向", "天候"] and pd.api.types.is_numeric_dtype(df_train[col]):
            df_train[col] = df_train[col].fillna(df_train[col].median())

    print(f"有効データ数: {len(df_train)}行")

    if len(df_train) == 0:
        print("エラー: 有効なデータ行が0件です。")
        return

    X = df_train[features]
    models = {}

    for i, target_col in enumerate(targets[:3], start=1):
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

    saved_package = {
        "models": models,
        "player_course_stats": None,
        "venue_wind_kimarite": venue_wind_kimarite,
        "player_fav_kimarite": None,
        "player_col": "選手名"
    }

    joblib.dump(saved_package, "boatrace_lgb_model.pkl")
    print("モデルと集計データを 'boatrace_lgb_model.pkl' に保存しました。")

if __name__ == "__main__":
    train_model()

