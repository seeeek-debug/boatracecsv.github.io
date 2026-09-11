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
        if "2026/01" in file or "2026/02" in file:
            continue
        if "2025" in file or "2024" in file:
            continue
        if "2026/" in file or "2026-" in file or any(f"2026{y}" in file for y in ["/03", "/04", "/05", "/06", "/07", "/08", "/09", "/10", "/11", "/12"]):
            target_files.append(file)
        elif "2026" in file:
            target_files.append(file)

    print(f"2026年3月以降の対象ファイル数: {len(target_files)}")

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

    # 級別の数値化
    if "級別" in df_train.columns:
        rank_map = {'A1': 4, 'A2': 3, 'B1': 2, 'B2': 1}
        df_train["級別"] = df_train["級別"].map(rank_map)

    player_col = None
    for col in ["選手コード", "登録番号", "選手名"]:
        if col in df_train.columns:
            player_col = col
            break

    # --- ★ 1. 過去レースから「選手別のコース実績・平均ST」を自動集計 ---
    player_course_stats = None
    if player_col and "枠番" in df_train.columns:
        print("過去レースの積み重ねから選手別の実績を計算中...")
        if "着順" in df_train.columns:
            df_train["is_win"] = (df_train["着順"] == 1).astype(int)
        else:
            df_train["is_win"] = 0

        if "スタートタイミング" in df_train.columns:
            df_train["スタートタイミング"] = pd.to_numeric(df_train["スタートタイミング"], errors='coerce')

        agg_dict = {}
        if "is_win" in df_train.columns:
            agg_dict["is_win"] = "mean"
        if "スタートタイミング" in df_train.columns:
            agg_dict["スタートタイミング"] = "mean"

        if agg_dict:
            player_course_stats = df_train.groupby([player_col, "枠番"]).agg(agg_dict).reset_index()
            player_course_stats = player_course_stats.rename(columns={
                "is_win": "実績_コース別勝率",
                "スタートタイミング": "実績_平均ST"
            })
            df_train = pd.merge(df_train, player_course_stats, on=[player_col, "枠番"], how="left")

    # --- ★ 2. 「決まり手」の高度な特徴量化 ---
    kimarite_col = None
    for col in ["決まり手", "決まり手 (逃げ・まくり等)"]:
        if col in df_train.columns:
            kimarite_col = col
            break

    venue_wind_kimarite = None
    player_fav_kimarite = None

    if kimarite_col:
        print("決まり手をエンコード・集計しています...")
        df_train["決まり手_コード"] = df_train[kimarite_col].astype('category')

        if "レース場" in df_train.columns and "風向" in df_train.columns:
            venue_wind_kimarite = df_train.groupby(["レース場", "風向"]).size().reset_index(name="場_風別_決まり手確率")
            df_train = pd.merge(df_train, venue_wind_kimarite, on=["レース場", "風向"], how="left")

        if player_col and "着順" in df_train.columns:
            winners = df_train[df_train["着順"] == 1]
            if len(winners) > 0 and kimarite_col in winners.columns:
                player_fav_kimarite = winners.groupby([player_col, kimarite_col]).size().reset_index(name="選手別_得意決まり手率")
                df_train = pd.merge(df_train, player_fav_kimarite, on=[player_col, kimarite_col], how="left")

    if player_col:
        df_train[player_col] = df_train[player_col].astype('category')

    # 特徴量リスト
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
        "回り足",
        "直線",
        "一周タイム",
        "半周タイム",
        "級別",
        "全国勝率",
        "当地勝率",
        "モーター2連率",
        "ボート2連率",
        "実績_コース別勝率",
        "実績_平均ST",
        "決まり手_コード",
        "場_風別_決まり手確率",
        "選手別_得意決まり手率",
    ]

    if player_col and player_col not in target_features:
        target_features.append(player_col)

    features = [col for col in target_features if col in df_train.columns]
    print(f"実際に使用する特徴量: {features}")

    target_candidates = [
        "1着_艇番", "2着_艇番", "3着_艇番",
        "4着_艇番", "5着_艇番", "6着_艇番"
    ]
    targets = [col for col in target_candidates if col in df_train.columns]

    for col in features:
        if col != player_col:
            df_train[col] = pd.to_numeric(df_train[col], errors='coerce')

    if len(df_train) > 100000:
        df_train = df_train.sample(n=100000, random_state=42)

    df_train = df_train.dropna(subset=targets + [c for c in features if c != player_col])
    print(f"有効データ数: {len(df_train)}行")

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

    # ====================================================
    # 🔥 【ここに追加】学習したモデルと集計データをまとめて保存する
    # ====================================================
    saved_package = {
        "models": models,
        "player_course_stats": player_course_stats,
        "venue_wind_kimarite": venue_wind_kimarite,
        "player_fav_kimarite": player_fav_kimarite,
        "player_col": player_col if player_col else "選手名"
    }

    joblib.dump(saved_package, "boatrace_lgb_model.pkl")
    print("モデルと集計データを 'boatrace_lgb_model.pkl' に保存しました。")

if __name__ == "__main__":
    train_model()

