import pandas as pd
import joblib

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
    
    year = target_race_code[:4]
    month = target_race_code[4:6]
    day = target_race_code[6:8]

    race_card_path = f"data/programs/race_cards/{year}/{month}/{day}.csv"
    sui_path = f"data/previews/sui/{year}/{month}/{day}.csv"
    orig_path = f"data/previews/original_exhibition/{year}/{month}/{day}.csv"

    print(f"出走表データを取得中: {race_card_path}")

    try:
        df_cards = pd.read_csv(race_card_path, dtype=str)
        df_cards.columns = df_cards.columns.str.strip()
    except Exception as e:
        print(f"出走表の読み込みに失敗しました: {e}")
        return

    # レースコードが一致する「1行（横持ち）」をピンポイントで検索
    df_matched = None
    for col in df_cards.columns:
        matched = df_cards[df_cards[col].str.strip() == str(target_race_code)]
        if len(matched) > 0:
            df_matched = matched.iloc[0:1]
            print(f"レースコード '{target_race_code}' を横持ちデータから発見しました。")
            break

    if df_matched is None or len(df_matched) == 0:
        print(f"エラー: レースコード '{target_race_code}' が見つかりませんでした。")
        return

    # 横持ちの1行（艇1〜艇6）を、モデルが処理しやすい6行の縦持ちデータに変換
    vertical_rows = []
    for i in range(1, 7):
        row_data = {}
        for col in df_matched.columns:
            if col.startswith(f"艇{i}_"):
                new_col = col.replace(f"艇{i}_", "")
                row_data[new_col] = df_matched[col].values[0]
            elif not col.startswith("艇"):
                row_data[col] = df_matched[col].values[0]
        row_data["枠番"] = i
        vertical_rows.append(row_data)

    df_target = pd.DataFrame(vertical_rows)
    print(f"縦持ちへの変換が完了しました（展開艇数: {len(df_target)}艇）")

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

    print(f"\n--- 予想結果（各着順の確率・艇番） ---")
    
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
