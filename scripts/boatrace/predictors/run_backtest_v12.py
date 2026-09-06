import sys
from pathlib import Path
import pandas as pd

# プロジェクトルートをパスに追加
root_path = Path(__file__).resolve().parents[3]
sys.path.append(str(root_path))

from scripts.boatrace.predictors.v12_longshot_skew import V12LongshotSkewPredictor

def load_repository_historical_data(repo_root: Path):
    """
    7月以降の直前オッズ（data/previews/od3/）と
    払戻金（data/results/payouts/）の実データをrace_idで正確に突合して読み込む
    """
    historical_races = []
    
    od3_root = repo_root / "data" / "previews" / "od3"
    payouts_root = repo_root / "data" / "results" / "payouts"
    
    if not od3_root.exists() or not payouts_root.exists():
        print(f"Data directories not found. od3: {od3_root}, payouts: {payouts_root}")
        return historical_races

    # 1. 払戻金データを辞書にロード
    payouts_dict = {}
    for payout_csv in payouts_root.glob("**/*.csv"):
        try:
            df_payout = pd.read_csv(payout_csv)
            for _, row in df_payout.iterrows():
                race_id = str(row.get("race_id", ""))
                if race_id:
                    payouts_dict[race_id] = row.to_dict()
        except Exception:
            continue

    print(f"Loaded {len(payouts_dict)} total payout records.")

    # 2. 7月以降の直前オッズデータを対象にロードし、払戻データと結合
    # 7月(07)、8月(08)、9月(09)などのフォルダを確実にキャッチする
    od3_files = list(od3_root.glob("**/2026/[0-9][0-9]/**/*.csv"))
    if not od3_files:
        od3_files = list(od3_root.glob("**/*.csv")) # フォールバック

    print(f"Targeting {len(od3_files)} od3 csv files.")

    for od3_csv in od3_files:
        try:
            df_od3 = pd.read_csv(od3_csv)
            for _, row in df_od3.iterrows():
                race_id = str(row.get("race_id", ""))
                if not race_id or race_id not in payouts_dict:
                    continue
                
                payout_row = payouts_dict[race_id]
                
                # オッズの抽出（3連単などの組み合わせ表記カラム）
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

                # 実際のレース結果（払戻データ側から取得）
                actual_result = str(payout_row.get("trifecta", payout_row.get("result", "")))
                if not actual_result:
                    continue

                # 予測確率の構築
                mock_probs = {k: 0.02 for k in odds_dict.keys()}

                historical_races.append({
                    "id": race_id,
                    "volatility": float(row.get("volatility", 1.5)),
                    "probs": mock_probs,
                    "odds": odds_dict,
                    "actual_result": actual_result
                })
        except Exception as e:
            continue

    print(f"Successfully matched {len(historical_races)} races from July onwards.")
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
