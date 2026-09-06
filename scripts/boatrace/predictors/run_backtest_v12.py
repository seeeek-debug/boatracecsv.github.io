import sys
from pathlib import Path
import pandas as pd

# プロジェクトルートをパスに追加
root_path = Path(__file__).resolve().parents[3]
sys.path.append(str(root_path))

from scripts.boatrace.predictors.v12_longshot_skew import V12LongshotSkewPredictor

def load_repository_historical_data(repo_root: Path):
    """
    data/estimate/ 内の既存プレディクター（v1_basic や v10_kimarite など）の
    データを読み込んでバックテスト用に変換する
    """
    historical_races = []
    estimate_root = repo_root / "data" / "estimate"
    
    if not estimate_root.exists():
        print(f"Estimate root not found: {estimate_root}")
        return historical_races

    # 存在する既存の予測データディレクトリを自動で探す
    source_dirs = [d for d in estimate_root.iterdir() if d.is_dir() and d.name not in ["stadium", "v12_longshot_skew"]]
    if not source_dirs:
        print("No existing estimate directories found.")
        return historical_races

    # 見つかった既存ディレクトリ（例: v1_basic や v10_kimarite）のデータを活用
    target_dir = source_dirs[0]
    print(f"Using existing data from: {target_dir.name}")

    for csv_path in sorted(target_dir.glob("**/*.csv")):
        try:
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                # リポジトリ内の既存CSV構造に合わせたマッピング
                race_id = str(row.get("race_id", row.get("date", "unknown")))
                
                # 既存データのスコアや確率、オッズ情報を抽出
                historical_races.append({
                    "id": race_id,
                    "volatility": float(row.get("volatility", 1.5)),
                    # 既存データのカラム構造に依存するため、必要に応じてキーを調整
                    "probs": row.to_dict(), 
                    "odds": {}, 
                    "actual_result": str(row.get("actual_result", ""))
                })
        except Exception as e:
            continue
            
    return historical_races

def main():
    repo_root = Path(__file__).resolve().parents[3]
    predictor = V12LongshotSkewPredictor()
    
    historical_data = load_repository_historical_data(repo_root)
    
    if not historical_data:
        print("No historical data could be loaded from repository.")
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
