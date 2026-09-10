import pandas as pd
import joblib
import glob
import numpy as np

def predict_race():
    model_filename = "boatrace_lgb_model.pkl"
    try:
        models = joblib.load(model_filename)
        print(f"モデル '{model_filename}' の読み込みに成功しました。")
    except Exception as e:
        print(f"モデルの読み込みに失敗しました: {e}")
        return

    # ★超重要：学習時に使われた正確な特徴量リストをモデルから直接取得する！
    if "rank_1" in models:
        expected_features = models["rank_1"].feature_name()
    else:
        print("エラー: モデル内に rank_1 が見つかりません。")
        return

    print(f"学習時の期待される特徴量数: {len(expected_features)}個")

    # テストデータの取得
    result_files = glob.glob("data/results/**/*.csv", recursive=True)
    if not result_files:
        result_files = glob.glob("data/**/*.csv", recursive=True)
        
    if not result_files:
        print("テスト用のCSVファイルが見つかりません。")
        return

    df_test = pd.read_csv(result_files[0])
    df_test.columns = df_test.columns.str.strip()
    
    if len(df_test) == 0:
        print("テストデータの行がありません。")
        return

    # 級別の数値化
    if "級別" in df_test.columns:
        rank_map = {'A1': 4, 'A2': 3, 'B1': 2, 'B2': 1}
        df_test["級別"] = df_test["級別"].map(rank_map)

    player_col = None
    for col in ["選手コード", "登録番号", "選手名"]:
        if col in df_test.columns:
            player_col = col
            break

    # 過去データの読み込み（統計用）
    df_all_list = []
    for f in result_files[:10]:
        try:
            temp_df = pd.read_csv(f)
            temp_df.columns = temp_df.columns.str.strip()
            df_all_list.append(temp_df)
        except:
            pass
            
    df_history = pd.concat(df_all_list, ignore_index=True) if df_all_list else df_test.copy()

    # 実績や決まり手確率の計算・マージ（学習時と同一ロジック）
    if player_col and "枠番" in df_history.columns and "着順" in df_history.columns:
        df_history["is_win"] = (df_history["着順"] == 1).astype(int)
        if "スタートタイミング" in df_history.columns:
            df_history["スタートタイミング"] = pd.to_numeric(df_history["スタートタイミング"], errors='coerce')

        player_course_stats = df_history.groupby([player_col, "枠番"]).agg({
            "is_win": "mean",
            "スタートタイミング": "mean"
        }).reset_index().rename(columns={
            "is_win": "実績_コース別勝率",
            "スタートタイミング": "実績_平均ST"
        })
        df_test = pd.merge(df_test, player_course_stats, on=[player_col, "枠番"], how="left")

    kimarite_col = None
    for col in ["決まり手", "決まり手（逃げ・まくり等）"]:
        if col in df_history.columns:
            kimarite_col = col
            break

    if kimarite_col and "レース場" in df_history.columns and "風向" in df_history.columns:
        venue_wind_kimarite = df_history.groupby(["レース場", "風向", kimarite_col]).size().reset_index(name="決まり手_発生回数")
        venue_wind_kimarite["場・風別_決まり手確率"] = venue_wind_kimarite["決まり手_発生回数"] / venue_wind_kimarite.groupby(["レース場", "風向"])["決まり手_発生回数"].transform("sum")
        df_test = pd.merge(df_test, venue_wind_kimarite[["レース場", "風向", kimarite_col, "場・風別_決まり手確率"]], on=["レース場", "風向", kimarite_col], how="left")

        winners = df_history[df_history["着順"] == 1]
        if len(winners) > 0 and kimarite_col in winners.columns and player_col:
            player_fav_kimarite = winners.groupby([player_col, kimarite_col]).size().reset_index(name="選手別_得意決まり手回数")
            player_fav_kimarite["選手別_得意決まり手率"] = player_fav_kimarite["選手別_得意決まり手回数"] / player_fav_kimarite.groupby(player_col)["選手別_得意決まり手回数"].transform("sum")
            df_test = pd.merge(df_test, player_fav_kimarite[[player_col, kimarite_col, "選手別_得意決まり手率"]], on=[player_col, kimarite_col], how="left")

    if player_col:
        df_test[player_col] = df_test[player_col].astype('category').cat.codes

    # 1行に絞る
    df_test = df_test.head(1).copy()

    # 数値変換
    for col in df_test.columns:
        if col != player_col:
            df_test[col] = pd.to_numeric(df_test[col], errors='coerce')

    # ★ここがミソ：学習時と完全に同じ特徴量リストに強制合わせ（足りない列は0で自動補完！）
    for col in expected_features:
        if col not in df_test.columns:
            df_test[col] = 0.0

    X_input = df_test[expected_features].fillna(0)

    print(f"\n--- 入力データ（特徴量数: {X_input.shape[1]}個で完全一致） ---")
    print(X_input)

    print("\n--- 予想結果（各着順の確率・艇番） ---")
    
    for i in range(1, 7):
        model_key = f"rank_{i}"
        if model_key in models:
            model = models[model_key]
            probs = model.predict(X_input)[0]
            predicted_boat = probs.argmax() + 1
            max_prob = probs.max() * 100
            print(f"{i}着予想: {predicted_boat}号艇 (確率: {max_prob:.1f}%)")

if __name__ == "__main__":
    predict_race()

