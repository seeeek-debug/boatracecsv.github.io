import sys
from pathlib import Path
import pandas as pd
import numpy as np

# LightGBMとscikit-learnのインポート（必要に応じて自動導入される前提）
try:
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split
except ImportError:
    pass

# プロジェクトルートをパスに追加
root_path = Path(__file__).resolve().parents[3]
sys.path.append(str(root_path))

from scripts.boatrace.predictors.v12_longshot_skew import V12LongshotSkewPredictor

def parse_class_rank(val):
    """選手級別を数値化する"""
    s = str(val).strip().upper()
    if "A1" in s: return 4
    if "A2" in s: return 3
    if "B1" in s: return 2
    if "B2" in s: return 1
    return 2 # デフォルトB1相当

def load_race_cards_dataset(repo_root: Path):
    """race_cardsフォルダから全レースの出走表データをロードして学習用・推論用に構造化する"""
    cards_root = repo_root / "data" / "programs" / "race_cards"
    card_files = list(cards_root.glob("**/*.csv"))
    
    races_data = {}
    ml_rows = []
    
    for c_file in card_files:
        try:
            df = pd.read_csv(c_file)
            for _, row in df.iterrows():
                rid = ""
                for col in ["レースコード", "race_id", "id", "RACE_ID"]:
                    if col in df.columns and pd.notna(row[col]):
                        rid = str(row[col]).strip()
                        break
                if not rid:
                    continue
                
                if rid not in races_data:
                    races_data[rid] = {}
                
                # 1〜6号艇のデータを抽出
                for boat_i in range(1, 7):
                    prefix = f"艇{boat_i}_"
                    # カラム名の揺れに対応
                    nat_win_col = next((c for c in [f"{prefix}全国勝率", f"boat_{boat_i}_national_win_rate"] if c in df.columns), None)
                    loc_win_col = next((c for c in [f"{prefix}当地勝率", f"boat_{boat_i}_local_win_rate"] if c in df.columns), None)
                    mot_2ren_col = next((c for c in [f"{prefix}モーター2連対率", f"boat_{boat_i}_motor_2ren"] if c in df.columns), None)
                    avg_st_col = next((c for c in [f"{prefix}平均ST", f"boat_{boat_i}_avg_st"] if c in df.columns), None)
                    class_col = next((c for c in [f"{prefix}級別", f"boat_{boat_i}_class"] if c in df.columns), None)
                    
                    try:
                        nat_win = float(row[nat_win_col]) if nat_win_col and pd.notna(row[nat_win_col]) else 5.0
                    except: nat_win = 5.0
                    
                    try:
                        loc_win = float(row[loc_win_col]) if loc_win_col and pd.notna(row[loc_win_col]) else 5.0
                    except: loc_win = 5.0
                    
                    try:
                        mot_2ren = float(row[mot_2ren_col]) if mot_2ren_col and pd.notna(row[mot_2ren_col]) else 30.0
                    except: mot_2ren = 30.0
                    
                    try:
                        avg_st = float(row[avg_st_col]) if avg_st_col and pd.notna(row[avg_st_col]) else 0.15
                    except: avg_st = 0.15
                    
                    class_val = parse_class_rank(row[class_col]) if class_col and pd.notna(row[class_col]) else 2
                    
                    boat_feat = {
                        "course": boat_i,
                        "national_win": nat_win,
                        "local_win": loc_win,
                        "motor_2ren": mot_2ren,
                        "avg_st": avg_st,
                        "class_val": class_val
                    }
                    races_data[rid][boat_i] = boat_feat
        except Exception:
            continue
            
    return races_data

def train_ml_win_model(races_data, payouts_dict):
    """出走表データと実際のレース結果からLightGBMで1着予測モデルを学習する"""
    X = []
    y = []
    
    for rid, boats in races_data.items():
        if rid not in payouts_dict:
            continue
        result_str = str(payouts_dict[rid]).strip()
        if not result_str or "-" not in result_str:
            try:
                actual_1st = int(result_str[0])
            except:
                continue
        else:
            try:
                actual_1st = int(result_str.split("-")[0])
            except:
                continue
                
        for boat_i, feat in boats.items():
            features = [
                feat["course"],
                feat["national_win"],
                feat["local_win"],
                feat["motor_2ren"],
                feat["avg_st"],
                feat["class_val"]
            ]
            X.append(features)
            y.append(1 if boat_i == actual_1st else 0)
            
    if not X:
        return None
        
    X = np.array(X)
    y = np.array(y)
    
    # LightGBMモデルの学習
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5,
        random_state=42,
        verbose=-1
    )
    model.fit(train_x, train_y)
    return model

def load_repository_historical_data(repo_root: Path):
    historical_races = []
    
    od3_root = repo_root / "data" / "previews" / "od3"
    payouts_root = repo_root / "data" / "results" / "payouts"
    
    # 1. 払戻金データをロード
    payouts_dict = {}
    payout_files = list(payouts_root.glob("**/*.csv"))
    
    for p_file in payout_files:
        try:
            df = pd.read_csv(p_file)
            for _, row in df.iterrows():
                rid = ""
                for col in ["レースコード", "race_id", "id", "RACE_ID"]:
                    if col in df.columns and pd.notna(row[col]):
                        rid = str(row[col]).strip()
                        break
                if not rid:
                    continue
                
                trifecta = ""
                for col in ["3連単_組番", "trifecta", "3rentan", "result"]:
                    if col in df.columns and pd.notna(row[col]):
                        trifecta = str(row[col]).strip()
                        break
                if trifecta:
                    payouts_dict[rid] = trifecta
        except Exception:
            continue

    # 2. 出走表データをロード
    races_data = load_race_cards_dataset(repo_root)
    
    # 3. 1着予測モデルの学習
    print("Training LightGBM Win Prediction Model using Race Cards & Results...")
    ml_model = train_ml_win_model(races_data, payouts_dict)
    if ml_model is None:
        print("Failed to train ML model.")
        return historical_races

    # 4. 直前オッズデータをロードして機械学習モデルによる予測とバックテスト用データを構築
    od3_files = list(od3_root.glob("**/*.csv"))
    
    matched_count = 0
    for od3_csv in od3_files:
        try:
            df_od3 = pd.read_csv(od3_csv)
            for _, row in df_od3.iterrows():
                rid = ""
                for col in ["レースコード", "race_id", "id", "RACE_ID"]:
                    if col in df_od3.columns and pd.notna(row[col]):
                        rid = str(row[col]).strip()
                        break
                
                if not rid or rid not in payouts_dict or rid not in races_data:
                    continue

                volatility = float(row.get("volatility", 1.5))
                boats = races_data[rid]
                
                # 機械学習モデルで各艇の1着確率を予測
                boat_win_probs = {}
                X_pred = []
                boat_indices = []
                for b_i in range(1, 7):
                    if b_i in boats:
                        feat = boats[b_i]
                        X_pred.append([
                            feat["course"],
                            feat["national_win"],
                            feat["local_win"],
                            feat["motor_2ren"],
                            feat["avg_st"],
                            feat["class_val"]
                        ])
                        boat_indices.append(b_i)
                
                if X_pred:
                    preds = ml_model.predict_proba(np.array(X_pred))[:, 1]
                    s = sum(preds)
                    if s > 0:
                        preds = preds / s
                    for idx, p in zip(boat_indices, preds):
                        boat_win_probs[idx] = max(p, 0.001)

                # ターゲット：中穴ゾーン（40倍〜200倍）
                raw_odds = {}
                for col in df_od3.columns:
                    if "-" in col:
                        clean_key = col.replace("3連単_", "").replace("3連複_", "").strip()
                        if "-" in clean_key:
                            try:
                                val = float(row[col])
                                if 40.0 <= val <= 200.0:
                                    raw_odds[clean_key] = val
                            except ValueError:
                                pass
                
                if not raw_odds:
                    continue

                # 3連単の確率を各艇の1着・2着・3着確率から算出（簡易マルコフ/順列モデル）
                probs = {}
                for k, o in raw_odds.items():
                    parts = k.split("-")
                    if len(parts) == 3:
                        try:
                            h1, h2, h3 = int(parts[0]), int(parts[1]), int(parts[2])
                            p1 = boat_win_probs.get(h1, 1/6)
                            # 残りの中での相対確率
                            p2 = boat_win_probs.get(h2, 1/6) / (1.0 - p1 + 1e-6)
                            p3 = boat_win_probs.get(h3, 1/6) / (1.0 - p1 - p2 + 1e-6)
                            base_p = max(p1 * p2 * p3, 1e-6)
                        except:
                            base_p = 1.0 / o
                    else:
                        base_p = 1.0 / o
                    
                    market_implied_p = 1.0 / o
                    probs[k] = base_p * 0.5 + market_implied_p * 0.5

                prob_sum = sum(probs.values())
                if prob_sum > 0:
                    probs = {k: p_val / prob_sum for k, p_val in probs.items()}

                # 期待値フィルター（EV >= 1.25）
                valid_bets = []
                for k, o in raw_odds.items():
                    ev = probs.get(k, 0) * o
                    if ev >= 1.25:
                        valid_bets.append((k, o, ev))
                
                if not valid_bets:
                    continue

                # 期待値順にソートし、上位1〜2点に厳選
                valid_bets.sort(key=lambda x: x[2], reverse=True)
                top_bets = valid_bets[:2]

                odds_dict = {k: o for k, o, ev in top_bets}
                filtered_probs = {k: probs[k] for k, o, ev in top_bets}

                if not odds_dict:
                    continue

                actual_result = str(payouts_dict[rid]).strip()

                historical_races.append({
                    "id": rid,
                    "volatility": volatility,
                    "probs": filtered_probs,
                    "odds": odds_dict,
                    "actual_result": actual_result
                })
                matched_count += 1
        except Exception:
            continue

    print(f"Successfully matched and filtered {len(historical_races)} races for backtest (ML LightGBM Model).")
    return historical_races

def main():
    repo_root = Path(__file__).resolve().parents[3]
    predictor = V12LongshotSkewPredictor()
    
    historical_data = load_repository_historical_data(repo_root)
    
    if not historical_data:
        print("No historical data could be loaded after ML filtering.")
        return
    
    print(f"=== V15 ML-Integrated Backtest Simulation ({len(historical_data)} races) ===")
    results = predictor.backtest_simulation(historical_data, initial_bankroll=1000000)
    
    print(f"初期資金: ¥{results['initial_bankroll']:,}")
    print(f"最終資金: ¥{results['final_bankroll']:,}")
    print(f"総投資額: ¥{results['total_investment']:,.2f}")
    print(f"総払戻金: ¥{results['total_payout']:,.2f}")
    print(f"回収率 (ROI): {results['roi']:.2f}%")
    print(f"最大ドローダウン (金額): ¥{results['max_drawdown']:,.2f}")
    print(f"最大ドローダウン (率): {results['max_drawdown_rate']:.2f}%")
    print("=== Backtest Finished Successfully ===")

if __name__ == "__main__":
    main()

