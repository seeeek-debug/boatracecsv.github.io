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

    target_race_code = "202609092010"
    target_date_str = "20260909"

    year = target_date_str[:4]
    month = target_date_str[4:6]
    day = target_date_str[6:8]

    print(f"レースコード '{target_race_code}' ({year}年{month}月{day}日) のデータを正確に探索中...")

    # フォルダ階層（YYYY/MM/DD.csv）にしっかり対応した検索パターン
    candidate_patterns = [
        f"**/{year}/{month}/{day}.csv",
        f"**/{year}/{month}/{int(day)}.csv",
        f"**/*{year}{month}{day}*.csv",
        f"**/realtime/{year}/{month}/{day}.csv",
        f"**/results/{year}/{month}/{day}.csv"
    ]
    
    candidate_files = []
    for pattern in candidate_patterns:
        candidate_files.extend(glob.glob(pattern, recursive=True))
    
    candidate_files = sorted(list(set(candidate_files)))
    
    # 予測出力やゴミファイルを除外
    candidate_files = [
        f for f in candidate_files 
        if "estimate" not in f 
        and "picks" not in f 
        and "payout" not in f 
        and "odds" not in f
    ]

    print(f"ヒットした候補ファイル数: {len(candidate_files)}件")

    df_test = None
    used_file = None

    # 1. 候補ファイルから該当レースコードを探索
    for f in candidate_files:
        try:
            temp_df = pd.read_csv(f)
            temp_df.columns = temp_df.columns.str.strip()
            for col in temp_df.columns:
                matched = temp_df[temp_df[col].astype(str) == str(target_race_code)]
                if len(matched) >= 4:
                    df_test = matched.copy()
                    used_file = f
                    break
            if df_test is not None:
                break
        except Exception as e:
            pass

    # 2. もし見つからない場合は、全CSVからそのレースコードを直接持っているファイルを強制検索（勝手に関係ない日を代用しない）
    if df_test is None:
        print("指定日のパスから直接見つからないため、全データからレースコードを総検索します...")
        all_files = [
            f for f in glob.glob("data/**/*.csv", recursive=True)
            if "estimate" not in f and "picks" not in f and "payout" not in f and "odds" not in f
        ]
        for f in all_files:
            try:
                temp_df = pd.read_csv(f)
                temp_df.columns = temp_df.columns.str.strip()
                for col in temp_df.columns:
                    matched = temp_df[temp_df[col].astype(str) == str(target_race_code)]
                    if len(matched) >= 4:
                        df_test = matched.copy()
                        used_file = f
                        break
                if df_test is not None:
                    break
            except:
                pass

    if df_test is None or len(df_test) == 0:
        print(f"エラー: レースコード '{target_race_code}' のデータがどのファイルにも見つかりませんでした。")
        return

    print(f"【使用ファイル】 {used_file} (取得艇数: {len(df_test)}艇)")

    # 級別の数値化
    if "級別" in df_test.columns:
        rank_map = {'A1': 4, 'A2': 3, 'B1': 2, 'B2': 1}
        df_test["級別"] = df_test["級別"].map(rank_map)

    player_col = None
    for col in ["選手コード", "登録番号", "選手名"]:
        if col in df_test.columns:
            player_col = col
            break

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

    print(f"\n--- 入力データ確認（全 {len(X_input)} 艇分） ---")
    print_cols = [c for c in ["レース場", "風速(m)", "風向", "全国勝率", "モーター2連率"] if c in X_input.columns]
    if print_cols:
        print(X_input[print_cols])
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
                print(f"  -> {boat_num}号艇のデータによる予測: 1番確率が高いのは {top_pred_boat}号艇 (確率: {max_prob:.1f}%)")

if __name__ == "__main__":
    predict_race()

