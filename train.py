import os
import glob
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import joblib

def load_and_merge_data():
    print("GitHub上の data/results/ からCSVデータを読み込んでいます...")
    
    result_files = glob.glob("data/results/**/*.csv", recursive=True)
    if not result_files:
        print("データファイルが見つかりません。パスを確認してください。")
        return None

    df_list = []
    for file in result_files:
        try:
            df = pd.read_csv(file)
            df_list.append(df)
        except Exception as e:
            print(f"ファイル読み込みエラー ({file}): {e}")
            
    if not df_list:
        return None
        
    df_base = pd.concat(df_list, ignore_index=True)
    return df_base

def train_model():
    df_train = load_and_merge_data()
    
    if df_train is None or len(df_train) == 0:
        print("有効な学習データがありません。処理を中断します。")
        return

    # 特徴量（説明変数）
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
    ]
    
    # 1着から6着までのターゲット列
    targets = [
        "1着_艇番", "2着_艇番", "3着_艇番", 
        "4着_艇番", "5着_艇番", "6着_艇番"
    ]

    # 必要な列に欠損がない行を抽出
    df_train = df_train.dropna(subset=targets + features)
    X = df_train[features]

    models = {}

    # 1着〜6着までそれぞれのモデルをループで学習
    for i, target_col in enumerate(targets, start=1):
        print(f"--- {i}着の予測モデルを学習中 ---")
        y = df_train[target_col].astype(int) - 1  # 艇番（1〜6）を0始まり（0〜5）に補正

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        params = {
            "objective": "multiclass",
            "num_class": 6,
            "metric": "multi_logloss",
            "boosting_type": "gbdt",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "random_state": 42
        }

        model = lgb.train(
            params,
            train_data,
            num_boost_round=300,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(30)]
        )
        
        models[f"rank_{i}"] = model

    # 1着〜6着すべてのモデルをまとめた辞書を保存
    model_filename = "boatrace_lgb_model.pkl"
    joblib.dump(models, model_filename)
    print(f"学習が完了しました！1〜6着の予測モデルをまとめて {model_filename} として保存しました。")

if __name__ == "__main__":
    train_model()
