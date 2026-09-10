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
        
    # 全体の過去データも読み込んで、学習時と同じように統計マスターを作る
    print("予測用の過去データを集計中...")
    df_all_list = []
    for f in result_files[:10]: # 直近のファイルから高速に集計
        try:
            temp_df = pd.read_csv(f)
            temp_df.columns = temp_df.columns.str.strip()
            df_all_list.append(temp_df)
        except:
            pass
            
    df_history = pd.concat(df_all_list, ignore_index=True) if df_all_list else df_test.copy()

    # 級別の数値化
    if "級別" in df_test.columns:
        rank_map = {'A1': 4, 'A2': 3, 'B1': 2, 'B2': 1}
        df_test["級別"] = df_test["級別"].map(rank_map)

    player_col = None
    for col in ["選手コード", "登録番号", "選手名"]:
        if col in df_test.columns:
            player_col = col
            break

    # -------------------------------------------------------------
    # ★ 学習時と同じロジックで「実績」や「決まり手確率」を計算して付与
    # -------------------------------------------------------------
    if player_col and "枠番" in df_history.columns and "着順" in df_history.columns:
        df_history["is_win"] = (df_history["着順"] == 1).astype(int)
        if "スタートタイミング" in df_history.columns:
            df_history["スタートタイミング"] = pd.to_numeric(df_history["スタートタイミング"], errors='coerce')

        # 選手×枠番ごとの実績
        player_course_stats = df_history.groupby([player_col, "枠番"]).agg({
            "is_win": "mean",
            "スタートタイミング": "mean"
        }).reset_index().rename(columns={
            "is_win": "実績_コース別勝率",
            "スタートタイミング": "実績_平均ST"
        })
        df_test = pd.merge(df_test, player_course_stats, on=[player_col, "枠番"], how="left")

    # 決まり手傾向の付与
    kimarite_col = None
    for col in ["決まり手", "決まり手（逃げ・まくり等）"]:
        if col in df_history.columns:
            kimarite_col = col
            break

    if kimarite_col and "レース場" in df_history.columns and "風向" in df_history.columns:
        # 場・風別の決まり手確率
        venue_wind_kimarite = df_history.groupby(["レース場", "風向", kimarite_col]).size().reset_index(name="決まり手_発生回数")
        venue_wind_kimarite["場・風別_決まり手確率"] = venue_wind_kimarite["決まり手_発生回数"] / venue_wind_kimarite.groupby(["レース場", "風向"])["決まり手_発生回数"].transform("sum")
        df_test = pd.merge(df_test, venue_wind_kimarite[["レース場", "風向", kimarite_col, "場・風別_決まり手確率"]], on=["レース場", "風向", kimarite_col], how="left")

        # 選手別の得意決まり手率
        winners = df_history[df_history["着順"] == 1]
        if len(winners) > 0 and kimarite_col in winners.columns and player_col:
            player_fav_kimarite = winners.groupby([player_col, kimarite_col]).size().reset_index(name="選手別_得意決まり手回数")
            player_fav_kimarite["選手別_得意決まり手率"] = player_fav_kimarite["選手別_得意決まり手回数"] / player_fav_kimarite.groupby(player_col)["選手別_得意決まり手回数"].transform("sum")
            df_test = pd.merge(df_test, player_fav_kimarite[[player_col, kimarite_col, "選手別_得意決まり手率"]], on=[player_col, kimarite_col], how="left")

    if player_col:
        df_test[player_col] = df_test[player_col].astype('category').cat.codes

    df_test = df_test.head(1).copy()

    # 学習時と同じ特徴量リスト（※未来の答えである '決まり手_コード' は除外）
    features = [
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
        "級別",
        "全国勝率",
        "当地勝率",
        "モーター2連率",
        "ボート2連率",
        "実績_コース別勝率",
        "実績_平均ST",
        "場・風別_決まり手確率",
        "選手別_得意決まり手率",
    ]
    
    if player_col and player_col not in features:
        features.append(player_col)

    use_features = [col for col in features if col in df_test.columns]
    
    for col in use_features:
        if col != player_col:
            df_test[col] = pd.to_numeric(df_test[col], errors='coerce')

    X_input = df_test[use_features].fillna(0)

    print("\n--- 入力データ（究極の学習モデル対応版） ---")
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
