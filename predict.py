import pandas as pd
import joblib
import numpy as np
import itertools

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

    def load_csv_safe(path):
        try:
            df = pd.read_csv(path, dtype=str)
            df.columns = df.columns.str.strip()
            return df
        except Exception:
            return None

    df_cards = load_csv_safe(race_card_path)
    df_sui = load_csv_safe(sui_path)
    df_orig = load_csv_safe(orig_path)

    if df_cards is None:
        print("エラー: 出走表データが取得できませんでした。")
        return

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

    base_info = {}
    for col in df_c_row.columns:
        if not col.startswith("艇"):
            base_info[col] = df_c_row[col].values[0]

    for df_r in [df_s_row, df_o_row]:
        if df_r is not None:
            for c in df_r.columns:
                if not c.startswith("艇"):
                    base_info[c] = df_r[c].values[0]

    vertical_rows = []
    for i in range(1, 7):
        row_data = base_info.copy()
        row_data["枠番"] = i
        for df_r in [df_c_row, df_s_row, df_o_row]:
            if df_r is not None:
                for col in df_r.columns:
                    if col.startswith(f"艇{i}_"):
                        row_data[col.replace(f"艇{i}_", "")] = df_r[col].values[0]
        vertical_rows.append(row_data)

    df_target = pd.DataFrame(vertical_rows)

    if "級別" in df_target.columns:
        rank_map = {'A1': 4, 'A2': 3, 'B1': 2, 'B2': 1}
        df_target["級別"] = df_target["級別"].map(rank_map)

    player_col = next((col for col in ["選手コード", "登録番号"] if col in df_target.columns), None)
    if player_col:
        df_target[player_col] = df_target[player_col].astype('category').cat.codes

    for col in df_target.columns:
        if col not in [player_col, "選手名", "支部", "出身地"]:
            df_target[col] = pd.to_numeric(df_target[col], errors='coerce')

    X_input = df_target.reindex(columns=expected_features, fill_value=0.0)

    # 各着順の確率マトリクスを格納 (index: 艇番1〜6, value: 確率配列)
    prob_matrix = {}
    for rank_idx, rank_name in enumerate(["rank_1", "rank_2", "rank_3"], 1):
        if rank_name in models:
            model = models[rank_name]
            # 各艇ごとの予測確率を取得
            preds_per_boat = []
            for idx, row in X_input.iterrows():
                p = model.predict(row.values.reshape(1, -1))[0]
                preds_per_boat.append(p)
            prob_matrix[rank_idx] = np.array(preds_per_boat)

    # 各艇ごとの1〜3着率を整理して表示
    print(f"\n--- 【各艇の着順確率一覧】 ---")
    boat_data = []
    for i in range(6):
        boat_num = i + 1
        name = df_target.loc[i, "選手名"] if "選手名" in df_target.columns else "不明"
        p1 = prob_matrix.get(1, np.zeros((6, 6)))[i][i] * 100 if 1 in prob_matrix else 0.0
        p2 = prob_matrix.get(2, np.zeros((6, 6)))[i][i] * 100 if 2 in prob_matrix else 0.0
        p3 = prob_matrix.get(3, np.zeros((6, 6)))[i][i] * 100 if 3 in prob_matrix else 0.0
        
        # モデルの出力構造（各艇に対する確率分布）に合わせた取得
        # ※もし model.predict が「その艇が各着順になる確率」を返す場合はインデックスを調整
        boat_data.append({"boat": boat_num, "name": name, "p1": p1, "p2": p2, "p3": p3})
        print(gh := f"  {boat_num}号艇 ({name}) -> 1着率: {p1:.1f}% | 2着率: {p2:.1f}% | 3着率: {p3:.1f}%")

    # 3連単の買い目計算（1着・2着・3着の確率を掛け合わせてスコア化）
    print(f"\n--- 【3連単 予想買い目（上位5点）】 ---")
    trifecta_scores = []
    
    # 簡易的に各モデルの確率表から上位の組み合わせを算出
    # (実際のマトリクスの形状に合わせてスコアリング)
    if 1 in prob_matrix and 2 in prob_matrix and 3 in prob_matrix:
        m1 = prob_matrix[1]
        m2 = prob_matrix[2]
        m3 = prob_matrix[3]
        
        for c1, c2, c3 in itertools.permutations(range(6), 3):
            # c1: 1着の艇index, c2: 2着の艇index, c3: 3着の艇index
            score = m1[c1][c1] * m2[c2][c2] * m3[c3][c3]
            trifecta_scores.append(((c1+1, c2+1, c3+1), score))
            
        trifecta_scores.sort(key=lambda x: x[1], reverse=True)
        
        for rank, (combo, score) in enumerate(trifecta_scores[:5], 1):
            print(f"  {rank}点目: {combo[0]} - {combo[1]} - {combo[2]} (期待度スコア: {score:.4f})")

    # レース展開の予想
    print(f"\n--- 【レース展開の考察】 ---")
    top_1st = max(boat_data, key=lambda x: x['p1'])
    top_2nd = max(boat_data, key=lambda x: x['p2'])
    print(f"  - イン優勢度および総合力から、{top_1st['boat']}号艇({top_1st['name']})軸のレース展開が濃厚。")
    print(f"  - 2着争いには機力と着率の良い {top_2nd['boat']}号艇({top_2nd['name']})が絡んでくる展開を推奨。")

if __name__ == "__main__":
    predict_race()

