import os
import glob
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import joblib

def load_and_merge_data():
    print("GitHub上のCSVデータを読み込んでいます...")
    
    # data/ 配下のすべてのCSVを自動で取得
    result_files = glob.glob("data/**/*.csv", recursive=True)
    if not result_files:
        print("データファイルが見つかりません。パスを確認してください。")
        return None

    df_list = []
    for file in result_files:
        try:
            df = pd.read_csv(file)
            # カラム名に入っている余計なスペースをすべて削除（これ重要！）
            df.columns = df.columns.str.strip()
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

    # 希望する特徴量候補
    target_features = [
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
    
    # 実際にデータフレームに存在するカラムだけに絞り込む（存在しないものでエラーになるのを防ぐ）
    features = [col for col in target_features if col in df_train.columns]
    print(f"実際に使用する特徴量: {features}")

    # 1着から6着までのターゲット列（こちらも存在するものを対象にする）
    target_candidates = [
        "1着_艇番", "2着_艇番", "3着_艇番", 
        "4着_艇番", "5着_艇番", "6着_艇番"
    ]
    targets = [col for col in target_candidates if col in df_train.columns]
    
    if not targets:
        print("エラー: 目的変数（着番データ）が見つかりません。")
        return

    # 必要な列に欠損がない行を抽出
    df_train = df_train.dropna(subset=targets + features)
    if len(df_train) == 0:
        print("エラー: 有効なデータ行が0件になってしまいました。")
        return

    X = df_train[features]
    models = {}

    # 1着〜6着までそれぞれのモデルをループで学習
    for i, target_col in enumerate(targets, start=1):
        print(f"--- {i}着の予測モデルを学習中 ({target_col}) ---")
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
    print(f"学習が完了しました！モデルを {model_filename} として保存しました。")

if __name__ == "__main__":
    train_model()
