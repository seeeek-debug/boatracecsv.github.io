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
        
    df_test = df_test.head(1).copy()

    if "級別" in df_test.columns:
        rank_map = {'A1': 4, 'A2': 3, 'B1': 2, 'B2': 1}
        df_test["級別"] = df_test["級別"].map(rank_map)

    player_col = None
    for col in ["選手コード", "登録番号", "選手名"]:
        if col in df_test.columns:
            player_col = col
            df_test[player_col] = df_test[player_col].astype('category').cat.codes
            break

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
    ]
    
    if player_col and player_col not in features:
        features.append(player_col)

    use_features = [col for col in features if col in df_test.columns]
    
    for col in use_features:
        if col != player_col:
            df_test[col] = pd.to_numeric(df_test[col], errors='coerce')

    X_input = df_test[use_features].fillna(0)

    print("\n--- 入力データ（選手個人の癖・スタート含む） ---")
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
