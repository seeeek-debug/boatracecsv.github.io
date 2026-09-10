import pandas as pd
import joblib

def predict_race():
    # 1. 保存したモデルの読み込み
    model_filename = "boatrace_lgb_model.pkl"
    try:
        models = joblib.load(model_filename)
        print(f"モデル '{model_filename}' の読み込みに成功しました。")
    except Exception as e:
        print(f"モデルの読み込みに失敗しました: {e}")
        return

    # 2. テスト用のレースデータ（例として最新のCSVから1行取得する想定）
    import glob
    result_files = glob.glob("data/results/**/*.csv", recursive=True)
    if not result_files:
        result_files = glob.glob("data/**/*.csv", recursive=True)
        
    if not result_files:
        print("テスト用のCSVファイルが見つかりません。")
        return

    # 適当なファイルから1行読み込んでテストデータを作成
    df_test = pd.read_csv(result_files[0]).dropna().head(1)
    df_test.columns = df_test.columns.str.strip()

    # 特徴量の定義（学習時と同じもの）
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
    ]

    # 存在する特徴量だけに絞る
    use_features = [col for col in features if col in df_test.columns]
    X_input = df_test[use_features]

    print("\n--- 入力データ（コンディション・展示） ---")
    print(X_input)

    print("\n--- 予想結果（各着順の確率・艇番） ---")
    
    # 1着〜6着のモデルを使ってそれぞれの確率を予測
    for i in range(1, 7):
        model_key = f"rank_{i}"
        if model_key in models:
            model = models[model_key]
            # 予測確率を取得 (0〜5のインデックスが艇番1〜6に対応)
            probs = model.predict(X_input)[0]
            
            # 最も確率が高い艇番（0〜5なので +1 する）
            predicted_boat = probs.argmax() + 1
            max_prob = probs.max() * 100
            
            print(f"{i}着予想: {predicted_boat}号艇 (確率: {max_prob:.1f}%)")

if __name__ == "__main__":
    predict_race()

