import sys
from pathlib import Path
import pandas as pd

# プロジェクトルートをパスに追加
root_path = Path(__file__).resolve().parents[3]
sys.path.append(str(root_path))

from scripts.boatrace.predictors.v12_longshot_skew import V12LongshotSkewPredictor

def load_stadium_win_rates(repo_root: Path):
    csv_path = repo_root / "data" / "estimate" / "stadium" / "course_win_rate.csv"
    win_rates = {}
    if not csv_path.exists():
        print(f"DEBUG: course_win_rate.csv not found at {csv_path}")
        return win_rates
    
    try:
        for enc in ["utf-8-sig", "utf-8", "cp932"]:
            try:
                df = pd.read_csv(csv_path, encoding=enc)
                break
            except Exception:
                continue
        else:
            df = pd.read_csv(csv_path, encoding="utf-8", errors="ignore")

        print(f"DEBUG: course_win_rate.csv columns: {list(df.columns)}")
        print(f"DEBUG: course_win_rate.csv total rows: {len(df)}")
        
        for _, row in df.iterrows():
            jyo_raw = row.get("場コード", "")
            if pd.isna(jyo_raw):
                continue
            jyo = str(jyo_raw).split(".")[0].strip().zfill(2)
            
            r_raw = None
            for col in ["レース図", "レース番号", "race_no", "r_no"]:
                if col in df.columns and pd.notna(row[col]):
                    r_raw = row[col]
                    break
            
            if r_raw is None:
                continue
            race_no = str(r_raw).split(".")[0].strip()
            
            key = f"{jyo}_{race_no}"
            rates = {}
            for c in range(1, 7):
                col_name = f"{c}コース勝率"
                if col_name in df.columns:
                    try:
                        rates[c] = float(row[col_name])
                    except:
                        rates[c] = 1.0 / 6.0
                else:
                    rates[c] = 1.0 / 6.0
            win_rates[key] = rates
            
        print(f"DEBUG: Successfully loaded stadium win rates for {len(win_rates)} keys.")
    except Exception as e:
        print(f"Error loading course_win_rate.csv: {e}")
    return win_rates

def parse_jyo_and_race(rid: str, row: pd.Series):
    jyo = ""
    race_no = ""
    for col in ["場コード", "jyo_cd", "stadium_code", "stadium"]:
        if col in row and pd.notna(row[col]):
            jyo = str(row[col]).split(".")[0].strip().zfill(2)
            break
    for col in ["レース図", "レース番号", "race_no", "r_no"]:
        if col in row and pd.notna(row[col]):
            race_no = str(row[col]).split(".")[0].strip()
            break
    
    if not jyo or not race_no:
        clean_id = ''.join(filter(str.isdigit, rid))
        if len(clean_id) >= 4:
            if not jyo:
                jyo = clean_id[-4:-2].zfill(2)
            if not race_no:
                try:
                    race_no = str(int(clean_id[-2:]))
                except:
                    race_no = clean_id[-2:]
    return jyo, race_no

def load_repository_historical_data(repo_root: Path):
    historical_races = []
    
    od3_root = repo_root / "data" / "previews" / "od3"
    payouts_root = repo_root / "data" / "results" / "payouts"
    
    stadium_win_rates = load_stadium_win_rates(repo_root)
    
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

    # 2. 直前オッズデータをロード
    od3_files = list(od3_root.glob("**/*.csv"))
    print(f"Found od3 files: {len(od3_files)}")
    
    matched_count = 0
    sample_checked = 0
    for od3_csv in od3_files:
        try:
            df_od3 = pd.read_csv(od3_csv)
            for _, row in df_od3.iterrows():
                rid = ""
                for col in ["レースコード", "race_id", "id", "RACE_ID"]:
                    if col in df_od3.columns and pd.notna(row[col]):
                        rid = str(row[col]).strip()
                        break
                
                if not rid:
                    continue
                
                if sample_checked < 3:
                    print(f"DEBUG OD3 row rid: {rid}, in payouts: {rid in payouts_dict}")
                    sample_checked += 1

                if rid not in payouts_dict:
                    continue

                volatility = float(row.get("volatility", 1.5))
                jyo, race_no = parse_jyo_and_race(rid, row)
                key = f"{jyo}_{race_no}"
                
                c_rates = stadium_win_rates.get(key, {i: 1.0/6.0 for i in range(1, 7)})

                # オッズ抽出：50倍〜250倍の穴ゾーン
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

                probs = {}
                for k, o in raw_odds.items():
                    parts = k.split("-")
                    if len(parts) == 3:
                        try:
                            h1, h2, h3 = int(parts[0]), int(parts[1]), int(parts[2])
                            p1 = c_rates.get(h1, 1/6)
                            p2 = c_rates.get(h2, 1/6) / (1.0 - p1 + 1e-6)
                            p3 = c_rates.get(h3, 1/6) / (1.0 - p1 - p2 + 1e-6)
                            base_p = max(p1 * p2 * p3, 1e-5)
                        except:
                            base_p = 1.0 / o
                    else:
                        base_p = 1.0 / o
                    
                    skew_factor = 1.0 + (o / 100.0) * (volatility / 2.0) * 0.15
                    probs[k] = base_p * skew_factor

                prob_sum = sum(probs.values())
                if prob_sum > 0:
                    probs = {k: p / prob_sum for k in probs.items()}

                odds_dict = {}
                filtered_probs = {}
                for k, o in raw_odds.items():
                    ev = probs.get(k, 0) * o
                    if ev >= 1.1:
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
        except Exception as e:
            print(f"Error parsing od3 file {od3_csv}: {e}")
            continue

    print(f"Successfully matched and filtered {len(historical_races)} races for backtest using stadium win rates.")
    return historical_races

def main():
    repo_root = Path(__file__).resolve().parents[3]
    predictor = V12LongshotSkewPredictor()
    
    historical_data = load_repository_historical_data(repo_root)
    
    if not historical_data:
        print("No historical data could be loaded after filtering.")
        return
    
    print(f"=== V12 Longshot Skew Backtest Simulation (Odds >= 50 with Stadium Data) ({len(historical_data)} races) ===")
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

