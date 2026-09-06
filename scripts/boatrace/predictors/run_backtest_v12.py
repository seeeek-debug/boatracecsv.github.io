import sys
from pathlib import Path
import pandas as pd
import json

# プロジェクトルートをパスに追加
root_path = Path(__file__).resolve().parents[3]
sys.path.append(str(root_path))

from scripts.boatrace.predictors.v12_longshot_skew import V12LongshotSkewPredictor

def load_repository_historical_data(repo_root: Path):
    """
    リポジトリ内のデータ構造（data/estimate や data/results 等）から
    過去の予測確率、オッズ、実際のレース結果をロードする
    """
    historical_races = []
    
    # 例: data/estimate/v12_longshot_skew/ 以下のCSVを走査して読み込む
    estimate_dir = repo_root / "data" / "estimate" / "v12_longshot_skew"
    if not estimate_dir.exists():
        print(f"Warning: Estimate directory not found: {estimate_dir}")
        return historical_races

    # 日付ごとのCSVファイルを探索
    for csv_path in sorted(estimate_dir.glob("**/*.csv")):
        df_pred = pd.read_csv(csv_path)
        
        # ここでリポジトリの既存のスキーマに合わせて確率・オッズ・結果を辞書形式にパースする
        # （実際のカラム名やデータ構造に合わせて適宜修正してください）
        for _, row in df_pred.iterrows():
            race_id = row.get("race_id")
            # 予測確率の復元（例: 120通りの確率がカラムまたはJSONとして保存されている場合）
            # probs = json.loads(row.get("probabilities", "{}"))
            # odds = json.loads(row.get("odds", "{}"))
            # actual_result = row.get("actual_result")
            
            # historical_races.append({
            #     "id": race_id,
            #     "volatility": row.get("volatility_score", 1.5),
            #     "probs": probs,
            #     "odds": odds,
            #     "actual_result": actual_result
            # })
            pass
            
    return historical_races

def main():
    repo_root = Path(__file__).resolve().parents[3]
    predictor = V12LongshotSkewPredictor()
    
    historical_data = load_repository_historical_data(repo_root)
    
    if not historical_data:
        print("No historical data loaded. Please check data paths or run data collection first.")
        return
    
    print(f"=== V12 Longshot Skew Backtest ({len(historical_data)} races) ===")
    results = predictor.backtest_simulation(historical_data, initial_bankroll=1000000)
    
    print(f"初期資金: ¥{results['initial_bankroll']:,}")
    print(f"最終資金: ¥{results['final_bankroll']:,}")
    print(f"総投資額: ¥{results['total_investment']:,.2f}")
    print(f"総払戻金: ¥{results['total_payout']:,.2f}")
    print(f"回収率 (ROI): {results['roi']:.2f}%")
    print(f"最大ドローダウン (金額): ¥{results['max_drawdown']:,.2f}")
    print(f"最大ドローダウン (率): {results['max_drawdown_rate']:.2f}%")

if __name__ == "__main__":
    main()

