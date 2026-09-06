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
    
    # 1. 払戻金データを日本語カラム名に合わせて確実にロード
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

    # 2. 直前オッズデータをロードして払戻データと結合
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

                # オッズ情報の抽出
                odds_dict = {}
                for col in df_od3.columns:
                    if "-" in col:
                        try:
                            val = float(row[col])
                            if val > 0:
                                odds_dict[col] = val
                        except ValueError:
                            pass
                
                if not odds_dict:
                    continue

                actual_result = payouts_dict[rid]

                historical_races.append({
                    "id": rid,
                    "volatility": float(row.get("volatility", 1.5)),
                    "probs": {k: 0.05 for k in odds_dict.keys()},
                    "odds": odds_dict,
                    "actual_result": actual_result
                })
                matched_count += 1
        except Exception:
            continue

    print(f"Successfully matched {len(historical_races)} races for backtest.")
    return historical_races

def main():
    repo_root = Path(__file__).resolve().parents[3]
    predictor = V12LongshotSkewPredictor()
    
    historical_data = load_repository_historical_data(repo_root)
    
    if not historical_data:
        print("No historical data could be loaded.")
        return
    
    print(f"=== V12 Longshot Skew Backtest Simulation ({len(historical_data)} races) ===")
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

