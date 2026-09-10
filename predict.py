import pandas as pd
import joblib
import glob

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

    target_race_code = "202609092010"
    
    # 重くなる原因になる全ファイルループは一切せず、dataフォルダ内のCSVを「1つだけ」安全に取得する
    target_files = [
        f for f in glob.glob("data/**/*.csv", recursive=True)
        if "estimate" not in f and "picks" not in f and "payout" not in f and "odds" not in f
    ]
    
    if not target_files:
        print("エラー: CSVファイルが見つかりません。")
        return

    # 9月9日のファイルがあれば最優先、なければ最初の1ファイルだけを選ぶ
    selected_file = None
    for f in target_files:
        if "2026/09/09" in f or "20260909" in f:
            selected_file = f
            break
    
    if not selected_file:
        selected_file = target_files[0]

    print(f"【超高速モード】ファイルを1つだけ読み込みます: {selected_file}")
    
    try:
        df_test = pd.read_csv(selected_file)
        df_test.columns = df_test.columns.str.strip()
    except Exception as e:
        print(f"ファイルの読み込みに失敗しました: {e}")
        return

    # 該当レースコードがあれば抽出、なければファイルの先頭6行を強制使用
    df_target = None
    for col in df_test.columns:
        matched = df_test[df_test[col].astype(str) == str(target_race_code)]
        if len(matched) >= 1:
            df_target = matched.copy()
            break

    if df_target is None or len(df_target) < 6:
        print("指定レースコードがこのファイルに見つからないため、ファイルの先頭6行を使用します。")
        df_target = df_test.head(6).copy()

    print(f"テストデータ行数: {len(df_target)}行")

    # 級別の数値化
    if "級別" in df_target.columns:
        rank_map = {'A1': 4, 'A2': 3, 'B1': 2, 'B2': 1}
        df_target["級別"] = df_target["級別"].map(rank_map)

    player_col = None
    for col in ["選手コード", "登録番号", "選手名"]:
        if col in df_target.columns:
            player_col = col
            break

    if player_col and player_col in df_target.columns:
        df_target[player_col] = df_target[player_col].astype('category').cat.codes

    # 数値変換
    for col in df_target.columns:
        if col != player_col:
            df_target[col] = pd.to_numeric(df_target[col], errors='coerce')

    # 特徴量リストの強制合わせ
    for col in expected_features:
        if col not in df_target.columns:
            df_target[col] = 0.0

    X_input = df_target[expected_features].fillna(0)

    print(f"\n--- 入力データ確認（全 {len(X_input)} 艇分） ---")
    print(X_input.head(2))

    print("\n--- 予想結果（各着順の確率・艇番） ---")
    
    for i in range(1, 7):
        model_key = f"rank_{i}"
        if model_key in models:
            model = models[model_key]
            print(f"\n【{i}着の予測】")
            for idx, row in X_input.iterrows():
                probs = model.predict(row.values.reshape(1, -1))[0]
                boat_num = df_target.loc[idx, "枠番"] if "枠番" in df_target.columns else idx + 1
                top_pred_boat = probs.argmax() + 1
                max_prob = probs.max() * 100
                print(f"  -> 艇番 {boat_num}: 最有力は {top_pred_boat}号艇 (確率: {max_prob:.1f}%)")

if __name__ == "__main__":
    predict_race()

