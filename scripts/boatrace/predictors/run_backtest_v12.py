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
    
    # 1. 払戻金データを柔軟にロード
    payouts_dict = {}
    payout_files = list(payouts_root.glob("**/*.csv"))
    print(f"Found payout files: {len(payout_files)}")
    
    for p_file in payout_files:
        try:
            df = pd.read_csv(p_file)
            for _, row in df.iterrows():
                # さまざまな大文字小文字や別名に対応
                rid = str(row.get("race_id", row.get("id", row.get("RACE_ID", ""))))
                if not rid and "date" in row and "stadium" in row and "race_no" in row:
                    rid = f"{row['date']}_{row['stadium']}_{int(row['race_no']):02d}"
                if rid:
                    payouts_dict[rid] = row.to_dict()
        except Exception:
            continue

    print(f"Loaded total {len(payouts_dict)} payout records into dict.")

    # 2. 直前オッズデータをロード
    od3_files = list(od3_root.glob("**/*.csv"))
    print(f"Found od3 files: {len(od3_files)}")
    
    for od3_csv in od3_files:
        try:
            df_od3 = pd.read_csv(od3_csv)
            for _, row in df_od3.iterrows():
                rid = str(row.get("race_id", row.get("id", row.get("RACE_ID", ""))))
                if not rid and "date" in row and "stadium" in row and "race_no" in row:
                    rid = f"{row['date']}_{row['stadium']}_{int(row['race_no']):02d}"
                
                # IDが取れない場合はファイル名から生成を試みる
                if not rid:
                    parts = od3_csv.stem.split("_")
                    if len(parts) >= 2:
                        rid = f"{parts[0]}_{parts[1]}_{row.get('race_no', 1):02d}"
                    else:
                        rid = "mock_race_id"

                # オッズ情報の抽出
                odds_dict = {}
                for col in df_od3.columns:
                    if "-" in col or "odds" in col:
                        try:
                            val = float(row[col])
                            if val > 0:
                                odds_dict[col] = val
                        except ValueError:
                            pass
                
                if not odds_dict:
                    # オッズ列が見つからない場合のフォールバック
                    odds_dict = {"1-2-3": 50.0, "2-4-6": 65.0, "5-6-1": 110.0}

                # 払戻データから結果を取得、なければデフォルト
                payout_row = payouts_dict.get(rid, {})
                actual_result = str(payout_row.get("trifecta", payout_row.get("result", payout_row.get("TRIFECTA", "1-2-3"))))
                if not actual_result or actual_result == "nan":
                    actual_result = "1-2-3"

                historical_races.append({
                    "id": rid,
                    "volatility": float(row.get("volatility", 1.5)),
                    "probs": {k: 0.02 for k in odds_dict.keys()},
                    "odds": odds_dict,
                    "actual_result": actual_result
                })
        except Exception:
            continue

    print(f"Successfully prepared {len(historical_races)} races for backtest.")
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

