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

    result_files = glob.glob("data/**/*.csv", recursive=True)
    if not result_files:
        print("テスト用のCSVファイルが見つかりません。")
        return

    # 払戻金ファイル（payoutsなど）を除外して、レース・出走データファイルを優先する
    target_files = [f for f in result_files if "payout" not in f and "odds" not in f]
    if not target_files:
        target_files = result_files

    print("全データから履歴と対象レースを探索中...")
    df_history_list = []
    df_test = None
    target_race_code = "202609092010" 

    for f in target_files:
        try:
            temp_df = pd.read_csv(f)
            temp_df.columns = temp_df.columns.str.strip()
            df_history_list.append(temp_df)
            
            for col in temp_df.columns:
                matched = temp_df[temp_df[col].astype(str) == str(target_race_code)]
                if len(matched) > 1:  # 1レース複数艇（通常6艇）あるはずなので1より大きいものを優先
                    df_test = matched.copy()
                    print(f"レースコード '{target_race_code}' を発見しました！（ファイル: {f}, 艇数: {len(df_test)}艇）")
                    break
            if df_test is not None and len(df_test) > 1:
                break
        except Exception as e:
            pass

    # もし見つからなければ、最初の有効そうなファイルから複数行（1レース分）を切り出す
    if df_test is None or len(df_test) <= 1:
        print(f"警告: 指定レースコードが見つからないため、代替データを使用します。")
        for f in target_files:
            try:
                fallback_df = pd.read_csv(f)
                fallback_df.columns = fallback_df.columns.str.strip()
                if len(fallback_df) >= 6:
                    df_test = fallback_df.head(6).copy()
                    print(f"代替ファイルを使用: {f} (6艇分)")
                    break
            except:
                pass

    if df_test is None or len(df_test) == 0:
        print("エラー: 有効なレースデータが取得できませんでした。")
        return

    df_history_all = pd.concat(df_history_list, ignore_index=True) if df_history_list else df_test.copy()

    # 級別の数値化
    if "級別" in df_test.columns:
        rank_map = {'A1': 4, 'A2': 3, 'B1': 2, 'B2': 1}
        df_test["級別"] = df_test["級別"].map(rank_map)

    player_col = None
    for col in ["選手コード", "登録番号", "選手名"]:
        if col in df_test.columns:
            player_col = col
            break

    # 過去データから実績やスタートタイミングの平均を計算してマージ
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
        if all(c in df_test.columns for c in [player_col, "枠番"]):
            df_test = pd.merge(df_test, player_course_stats, on=[player_col, "枠番"], how="left")

    # 決まり手確率の計算・マージ
    kimarite_col = None
    for col in ["決まり手", "決まり手（逃げ・まくり等）"]:
        if col in df_history_all.columns:
            kimarite_col = col
            break

    if kimarite_col and "レース場" in df_history_all.columns and "風向" in df_history_all.columns and "風向" in df_test.columns:
        venue_wind_kimarite = df_history_all.groupby(["レース場", "風向", kimarite_col]).size().reset_index(name="決まり手_発生回数")
        venue_wind_kimarite["場・風別_決まり手確率"] = venue_wind_kimarite["決まり手_発生回数"] / venue_wind_kimarite.groupby(["レース場", "風向"])["決まり手_発生回数"].transform("sum")
        if all(c in df_test.columns for c in ["レース場", "風向", kimarite_col]):
            df_test = pd.merge(df_test, venue_wind_kimarite[["レース場", "風向", kimarite_col, "場・風別_決まり手確率"]], on=["レース場", "風向", kimarite_col], how="left")

        winners = df_history_all[df_history_all["着順"] == 1]
        if len(winners) > 0 and kimarite_col in winners.columns and player_col:
            player_fav_kimarite = winners.groupby([player_col, kimarite_col]).size().reset_index(name="選手別_得意決まり手回数")
            player_fav_kimarite["選手別_得意決まり手率"] = player_fav_kimarite["選手別_得意決まり手回数"] / player_fav_kimarite.groupby(player_col)["選手別_得意決まり手回数"].transform("sum")
            if all(c in df_test.columns for c in [player_col, kimarite_col]):
                df_test = pd.merge(df_test, player_fav_kimarite[[player_col, kimarite_col, "選手別_得意決まり手率"]], on=[player_col, kimarite_col], how="left")

    if player_col and player_col in df_test.columns:
        df_test[player_col] = df_test[player_col].astype('category').cat.codes

    # 数値変換
    for col in df_test.columns:
        if col != player_col:
            df_test[col] = pd.to_numeric(df_test[col], errors='coerce')

    # 特徴量リストの強制合わせ
    for col in expected_features:
        if col not in df_test.columns:
            df_test[col] = 0.0

    X_input = df_test[expected_features].fillna(0)

    print(f"\n--- 入力データ（対象レース / 艇数: {len(X_input)}艇） ---")
    # 存在するカラムだけを安全に抽出して表示
    print_cols = [c for c in ["レース場", "風速(m)", "風向", "全国勝率", "モーター2連率"] if c in X_input.columns]
    if print_cols:
        print(X_input[print_cols].head(2))
    else:
        print(X_input.head(2))

    print("\n--- 予想結果（各着順の確率・艇番） ---")
    
    for i in range(1, 7):
        model_key = f"rank_{i}"
        if model_key in models:
            model = models[model_key]
            print(f"\n【{i}着の予測】")
            for idx, row in X_input.iterrows():
                probs = model.predict(row.values.reshape(1, -1))[0]
                boat_num = df_test.loc[idx, "枠番"] if "枠番" in df_test.columns else idx + 1
                top_pred_boat = probs.argmax() + 1
                max_prob = probs.max() * 100
                print(f"  -> 艇番 {boat_num}: この着順の最有力は {top_pred_boat}号艇 (確率: {max_prob:.1f}%)")

if __name__ == "__main__":
    predict_race()

