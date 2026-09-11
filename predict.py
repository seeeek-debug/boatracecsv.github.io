import os
import glob
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib

def load_prediction_data():
    print("推論用データの読み込みを開始します...")
    result_files = glob.glob("data/results/**/*.csv", recursive=True)
    if not result_files:
        result_files = glob.glob("data/**/*.csv", recursive=True)

    if not result_files:
        print("エラー: 推論用データファイルが見つかりません。")
        return None

    # 最新のファイル、または直近の対象ファイルを選択
    target_file = sorted(result_files)[-1]
    print(f"対象ファイル: {target_file}")

    try:
        df = pd.read_csv(target_file)
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        print(f"ファイル読み込みエラー ({target_file}): {e}")
        return None

def predict_races():
    model_path = "boatrace_lgb_model.pkl"
    if not os.path.exists(model_path):
        print(f"エラー: モデルファイル '{model_path}' が見つかりません。先に学習スクリプトを実行してください。")
        return

    print("学習済みモデルと集計データをロードしています...")
    package = joblib.load(model_path)
    models = package["models"]
    venue_wind_kimarite = package["venue_wind_kimarite"]

    df = load_prediction_data()
    if df is None or len(df) == 0:
        print("有効な推論データがありません。")
        return

    # 学習時と同じ決まり手のエンコード・集計ロジック
    kimarite_col = None
    for col in ["決まり手", "決まり手 (逃げ・まくり等)"]:
        if col in df.columns:
            kimarite_col = col
            break

    if kimarite_col and kimarite_col in df.columns:
        df["決まり手_コード"] = df[kimarite_col].astype('category')
    else:
        df["決まり手_コード"] = pd.Categorical([np.nan] * len(df))

    if venue_wind_kimarite is not None and "レース場" in df.columns and "風向" in df.columns:
        df = pd.merge(df, venue_wind_kimarite, on=["レース場", "風向"], how="left")
    else:
        df["場_風別_決まり手確率"] = np.nan

    # 学習時と完全に一致させた特徴量リスト
    target_features = [
        "レース場",
        "風速(m)",
        "波の高さ(cm)",
        "水温(℃)",
        "気温(℃)",
        "風向",
        "天候",
        "1コース_スタートタイミング",
        "2コース_スタートタイミング",
        "3コース_スタートタイミング",
        "4コース_スタートタイミング",
        "5コース_スタートタイミング",
        "6コース_スタートタイミング",
        "場_風別_決まり手確率",
        "決まり手_コード"
    ]

    features = [col for col in target_features if col in df.columns]
    print(f"使用する特徴量: {features}")

    # 数値変換
    for col in features:
        if col not in ["決まり手_コード", "レース場", "風向", "天候"]:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 特徴量の欠損値補完（学習時と同様の安全性確保）
    for col in features:
        if col not in ["決まり手_コード", "レース場", "風向", "天候"] and pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(0)

    X = df[features]

    print("1着〜3着の予測を実行中...")
    for i in range(1, 4):
        rank_key = f"rank_{i}"
        if rank_key in models:
            model = models[rank_key]
            # 多クラス分類の確率を出力 (shape: [n_samples, 6])
            probs = model.predict(X)
            # 最も確率の高い艇番（0〜5 なので +1 して 1〜6艇番に変換）
            pred_tban = np.argmax(probs, axis=1) + 1
            df[f"予測_{i}着"] = pred_tban
            df[f"予測_{i}着_確率"] = np.max(probs, axis=1)

    # 出力列の整理
    race_id_col = "レースコード" if "レースコード" in df.columns else (df.columns[0] if len(df.columns) > 0 else None)
    
    output_cols = []
    if race_id_col:
        output_cols.append(race_id_col)
    if "レース場" in df.columns:
        output_cols.append("レース場")
    if "レース回" in df.columns:
        output_cols.append("レース回")

    output_cols.extend([
        "予測_1着", "予測_1着_確率", 
        "予測_2着", "予測_2着_確率", 
        "予測_3着", "予測_3着_確率"
    ])
    
    existing_output_cols = [c for c in output_cols if c in df.columns]

    print("\n=== 予測結果サンプル ===")
    print(df[existing_output_cols].head(10))

    output_filename = "boatrace_prediction_results.csv"
    df[existing_output_cols].to_csv(output_filename, index=False, encoding="utf-8-sig")
    print(f"\nすべての予測結果を '{output_filename}' に保存しました。")

if __name__ == "__main__":
    predict_races()

