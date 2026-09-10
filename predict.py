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

    print(f"必要なファイルを一括取得中...")

    # 各CSVの読み込み関数
    def load_csv_safe(path):
        try:
            df = pd.read_csv(path, dtype=str)
            df.columns = df.columns.str.strip()
            return df
        except Exception as e:
            print(f"ファイルの読み込みに失敗しました ({path}): {e}")
            return None

    df_cards = load_csv_safe(race_card_path)
    df_sui = load_csv_safe(sui_path)
    df_orig = load_csv_safe(orig_path)

    if df_cards is None:
        print("エラー: 出走表データが取得できないため処理を中断します。")
        return

    # 指定したレースコードの行を横持ちのまま1行だけ抜き出すヘルパー
    def get_matched_row(df, code):
        if df is None: return None
        for col in df.columns:
            matched = df[df[col].str.strip() == str(code)]
            if len(matched) > 0:
                return matched.iloc[0:1].copy()
        return None

    df_c_row = get_matched_row(df_cards, target_race_code)
    if df_c_row is None or len(df_c_row) == 0:
        print(f"エラー: レースコード '{target_race_code}' が出走表に見つかりませんでした。")
        return

    df_s_row = get_matched_row(df_sui, target_race_code)
    df_o_row = get_matched_row(df_orig, target_race_code)

    # 横方向へデータを結合
    combined_row = df_c_row.reset_index(drop=True)
    
    if df_s_row is not None:
        for c in df_s_row.columns:
            if c not in combined_row.columns:
                combined_row[f"sui_{c}"] = df_s_row[c].values[0]

    if df_o_row is not None:
        for c in df_o_row.columns:
            if c not in combined_row.columns:
                combined_row[f"orig_{c}"] = df_o_row[c].values[0]

    # 1行（横持ち）を6艇分（縦持ち）に展開
    vertical_rows = []
    for i in range(1, 7):
        row_data = {}
        for col in combined_row.columns:
            if col.startswith(f"艇{i}_"):
                new_col = col.replace(f"艇{i}_", "")
                row_data[new_col] = combined_row[col].values[0]
            elif not col.startswith("艇"):
                row_data[col] = combined_row[col].values[0]
        row_data["枠番"] = i
        vertical_rows.append(row_data)

    df_target = pd.DataFrame(vertical_rows)
    print(f"データの結合と縦持ち展開が完了しました（展開艇数: {len(df_target)}艇）")

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

