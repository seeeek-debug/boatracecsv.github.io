import sys
from pathlib import Path
import pandas as pd

# プロジェクトルートをパスに追加
root_path = Path(__file__).resolve().parents[3]
sys.path.append(str(root_path))

from scripts.boatrace.predictors.v12_longshot_skew import V12LongshotSkewPredictor

def load_repository_historical_data(repo_root: Path):
    historical_races = []
    
    od3_root = repo_root / "data" / "previews" / "od3"
    payouts_root = repo_root / "data" / "results" / "payouts"
    
    # 1. 払戻金データをロード
    payouts_dict = {}
    payout_files = list(payouts_root.glob("**/*.csv"))
    print(f"Found payout files: {len(payout_files)}")
    
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

    print(f"Loaded total {len(payouts_dict)} payout records into dict.")

    # 2. 直前オッズデータをロードして高度なフィルタリングと確率モデルを適用
    od3_files = list(od3_root.glob("**/*.csv"))
    print(f"Found od3 files: {len(od3_files)}")
    
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
                
                if not rid or rid not in payouts_dict:
                    continue

                volatility = float(row.get("volatility", 1.5))

                # オッズ情報の抽出と【オッズのフィルタリング】（50倍〜250倍の穴ゾーンに限定）
                raw_odds = {}
                for col in df_od3.columns:
                    if "-" in col:
                        clean_key = col.replace("3連単_", "").replace("3連複_", "").strip()
                        if "-" in clean_key:
                            try:
                                val = float(row[col])
                                if 50.0 <= val <= 250.0:
                                    raw_odds[clean_key] = val
                            except ValueError:
                                pass
                
                if not raw_odds:
                    continue

                # 【確率計算モデルの精度調整】
                total_inv_odds = sum(1.0 / o for o in raw_odds.values())
                probs = {}
                for k, o in raw_odds.items():
                    implied_prob = (1.0 / o) / total_inv_odds
                    skew_factor = 1.0 + (o / 100.0) * (volatility / 2.0) * 0.15
                    probs[k] = implied_prob * skew_factor

                # 確率の正規化
                prob_sum = sum(probs.values())
                if prob_sum > 0:
                    probs = {k: p / prob_sum for k in probs.items()}

                # 【期待値（EV）の閾値フィルタリング】（EV >= 1.2）
                odds_dict = {}
                filtered_probs = {}
                for k, o in raw_odds.items():
                    ev = probs.get(k, 0) * o
                    if ev >= 1.2:
                        odds_dict[k] = o
                        filtered_probs[k] = probs[k]

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

    print(f"Successfully matched and filtered {len(historical_races)} races for backtest.")
    return historical_races

def main():
    repo_root = Path(__file__).resolve().parents[3]
    predictor = V12LongshotSkewPredictor()
    
    historical_data = load_repository_historical_data(repo_root)
    
    if not historical_data:
        print("No historical data could be loaded after filtering.")
        return
    
    print(f"=== V12 Longshot Skew Backtest Simulation (Odds >= 50) ({len(historical_data)} races) ===")
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

