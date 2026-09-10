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

    if "rank_1" in models:
        expected_features = models["rank_1"].feature_name()
    else:
        print("エラー: モデル内に rank_1 が見つかりません。")
        return

    # テストデータの取得
    result_files = glob.glob("data/results/**/*.csv", recursive=True)
    if not result_files:
        result_files = glob.glob("data/**/*.csv", recursive=True)
        
    if not result_files:
        print("テスト用のCSVファイルが見つかりません。")
        return

    # 全ファイルを読み込んで「場コード20」かつ「10R」のデータを探す
    print("指定されたレース（場コード20の10R）を探索中...")
    target_row = None
    df_history = []

    for f in result_files:
        try:
            temp_df = pd.read_csv(f)
            temp_df.columns = temp_df.columns.str.strip()
            df_history.append(temp_df)
            
            # 場コード20かつ10Rの行があるかチェック
            # （※カラム名は「レース場」「R」「レース」などの可能性があるので柔軟に探す）
            venue_col = next((c for c in ["レース場", "場コード", "場"] if c in temp_df.columns), None)
            race_col = next((c for c in ["R", "レース", "レース番号"] if c in temp_df.columns), None)
            
            if venue_col and race_col:
                # 20 と 10（または "10R"）で絞り込み
                matched = temp_df[
                    (temp_df[venue_col].astype(str).str.contains("20")) & 
                    (temp_df[race_col].astype(str).str.contains("10"))
                ]
                if len(matched) > 0:
                    df_test = matched.copy()
                    print(f"該当レースを発見しました！（ファイル: {f}, 行数: {len(df_test)}行）")
                    break
        except Exception as e:
            pass

    # もし見つからんかったら、安全のために最初のファイルをフォールバック
    if 'df_test' not in locals() or len(df_test) == 0:
        print("警告: 指定条件のレースが見つからなかったため、先頭のデータを使用します。")
        df_test = pd.read_csv(result_files[0])
        df_test.columns = df_test.columns.str.strip()

    df_history_all = pd.concat(df_history, ignore_index=True) if df_history else df_test.copy()

    # 級別の数値化
    if "級別" in df_test.columns:
        rank_map = {'A1': 4, 'A2': 3, 'B1': 2, 'B2': 1}
        df_test["級別"] = df_test["級別"].map(rank_map)

    player_col = None
    for col in ["選手コード", "登録番号", "選手名"]:
        if col in df_test.columns:
            player_col = col
            break

    # 実績や決まり手確率の計算・マージ
    if player_col and "枠番" in df_history_all.columns and "着順" in df_history_all.columns:
        df_history_all["is_win"] = (df_history_all["着順"] == 1).astype(int)
        if "スタートタイミング" in df_history_all.columns:
            df_history_all["スタートタイミング"] = pd.to_numeric(df_history_all["スタートタイミング"], errors='coerce')

        player_course_stats = df_history_all.groupby([player_col, "枠番"]).agg({
            "is_win": "mean",
            "スタートタイミング": "mean"
        }).reset_index().rename(columns={
            "is_win": "実績_コース別勝率",
            "スタートタイミング": "実績_平均ST"
        })
        df_test = pd.merge(df_test, player_course_stats, on=[player_col, "枠番"], how="left")

    kimarite_col = None
    for col in ["決まり手", "決まり手（逃げ・まくり等）"]:
        if col in df_history_all.columns:
            kimarite_col = col
            break

    if kimarite_col and "レース場" in df_history_all.columns and "風向" in df_history_all.columns:
        venue_wind_kimarite = df_history_all.groupby(["レース場", "風向", kimarite_col]).size().reset_index(name="決まり手_発生回数")
        venue_wind_kimarite["場・風別_決まり手確率"] = venue_wind_kimarite["決まり手_発生回数"] / venue_wind_kimarite.groupby(["レース場", "風向"])["決まり手_発生回数"].transform("sum")
        df_test = pd.merge(df_test, venue_wind_kimarite[["レース場", "風向", kimarite_col, "場・風別_決まり手確率"]], on=["レース場", "風向", kimarite_col], how="left")

        winners = df_history_all[df_history_all["着順"] == 1]
        if len(winners) > 0 and kimarite_col in winners.columns and player_col:
            player_fav_kimarite = winners.groupby([player_col, kimarite_col]).size().reset_index(name="選手別_得意決まり手回数")
            player_fav_kimarite["選手別_得意決まり手率"] = player_fav_kimarite["選手別_得意決まり手回数"] / player_fav_kimarite.groupby(player_col)["選手別_得意決まり手回数"].transform("sum")
            df_test = pd.merge(df_test, player_fav_kimarite[[player_col, kimarite_col, "選手別_得意決まり手率"]], on=[player_col, kimarite_col], how="left")

    if player_col:
        df_test[player_col] = df_test[player_col].astype('category').cat.codes

    # 数値変換
    for col in df_test.columns:
        if col != player_col:
            df_test[col] = pd.to_numeric(df_test[col], errors='coerce')

    # 特徴量リストの強制合わせ
    for col in expected_features:
        if col not in df_test.columns:
            df_test[col] = 0.0

    # 1行だけでなく、そのレースの全艇分（または先頭）をしっかり確認できるようにする
    X_input = df_test[expected_features].fillna(0)

    print(f"\n--- 入力データ（場20・10R / 特徴量数: {X_input.shape[1]}個） ---")
    print(X_input.head())

    print("\n--- 予想結果（各着順の確率・艇番） ---")
    
    for i in range(1, 7):
        model_key = f"rank_{i}"
        if model_key in models:
            model = models[model_key]
            # 複数艇ある場合は先頭の艇、あるいは全艇分の予測を出す
            probs = model.predict(X_input)[0]
            predicted_boat = probs.argmax() + 1
            max_prob = probs.max() * 100
            print(f"{i}着予想: {predicted_boat}号艇 (確率: {max_prob:.1f}%)")

if __name__ == "__main__":
    predict_race()

