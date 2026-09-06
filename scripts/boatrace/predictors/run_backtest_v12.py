import sys
from pathlib import Path
import pandas as pd
import numpy as np

root_path = Path(__file__).resolve().parents[3]
sys.path.append(str(root_path))

from scripts.boatrace.predictors.v12_longshot_skew import V12LongshotSkewPredictor

def parse_class_rank(val):
    """選手級別を数値化する"""
    s = str(val).strip().upper()
    if "A1" in s: return 4.0
    if "A2" in s: return 3.0
    if "B1" in s: return 2.0
    if "B2" in s: return 1.0
    return 2.0

def load_sui_dataset(repo_root: Path):
    """水面コンディション（波高）をロードする"""
    sui_root = repo_root / "data" / "previews" / "sui"
    sui_data = {}
    for f in sui_root.glob("**/*.csv"):
        try:
            df = pd.read_csv(f)
            for _, row in df.iterrows():
                rid = ""
                for col in ["レースコード", "race_id", "id", "RACE_ID"]:
                    if col in df.columns and pd.notna(row[col]):
                        rid = str(row[col]).strip()
                        break
                if not rid: continue
                wave_height = 1.0
                for col in ["波の高さ(cm)", "wave_height", "波高"]:
                    if col in df.columns and pd.notna(row[col]):
                        try: wave_height = float(row[col]); break
                        except: pass
                sui_data[rid] = {"wave_height": wave_height}
        except Exception:
            continue
    return sui_data

def load_original_exhibition_dataset(repo_root: Path):
    """オリジナル展示データ（展示タイム）をロードする"""
    ex_root = repo_root / "data" / "previews" / "original_exhibition"
    ex_data = {}
    for f in ex_root.glob("**/*.csv"):
        try:
            df = pd.read_csv(f)
            for _, row in df.iterrows():
                rid = ""
                for col in ["レースコード", "race_id", "id", "RACE_ID"]:
                    if col in df.columns and pd.notna(row[col]):
                        rid = str(row[col]).strip()
                        break
                if not rid: continue
                if rid not in ex_data: ex_data[rid] = {}
                for boat_i in range(1, 7):
                    prefix = f"艇{boat_i}_"
                    time_val = 6.8
                    for c in [f"{prefix}展示タイム", f"{prefix}タイム", f"boat_{boat_i}_ex_time"]:
                        if c in df.columns and pd.notna(row[c]):
                            try: time_val = float(row[c]); break
                            except: pass
                    ex_data[rid][boat_i] = {"ex_time": time_val}
        except Exception:
            continue
    return ex_data

def load_race_cards_dataset(repo_root: Path):
    """出走表データ（勝率・モーター・ST・級別）をロードする"""
    cards_root = repo_root / "data" / "programs" / "race_cards"
    races_data = {}
    for c_file in cards_root.glob("**/*.csv"):
        try:
            df = pd.read_csv(c_file)
            for _, row in df.iterrows():
                rid = ""
                for col in ["レースコード", "race_id", "id", "RACE_ID"]:
                    if col in df.columns and pd.notna(row[col]):
                        rid = str(row[col]).strip()
                        break
                if not rid: continue
                if rid not in races_data: races_data[rid] = {}
                for boat_i in range(1, 7):
                    prefix = f"艇{boat_i}_"
                    nat_win_col = next((c for c in [f"{prefix}全国勝率", f"boat_{boat_i}_national_win_rate"] if c in df.columns), None)
                    loc_win_col = next((c for c in [f"{prefix}當地勝率", f"boat_{boat_i}_local_win_rate"] if c in df.columns), None)
                    mot_2ren_col = next((c for c in [f"{prefix}モーター2連対率", f"boat_{boat_i}_motor_2ren"] if c in df.columns), None)
                    avg_st_col = next((c for c in [f"{prefix}平均ST", f"boat_{boat_i}_avg_st"] if c in df.columns), None)
                    class_col = next((c for c in [f"{prefix}級別", f"boat_{boat_i}_class"] if c in df.columns), None)
                    
                    try: nat_win = float(row[nat_win_col]) if nat_win_col and pd.notna(row[nat_win_col]) else 5.0
                    except: nat_win = 5.0
                    try: loc_win = float(row[loc_win_col]) if loc_win_col and pd.notna(row[loc_win_col]) else 5.0
                    except: loc_win = 5.0
                    try: mot_2ren = float(row[mot_2ren_col]) if mot_2ren_col and pd.notna(row[mot_2ren_col]) else 30.0
                    except: mot_2ren = 30.0
                    try: avg_st = float(row[avg_st_col]) if avg_st_col and pd.notna(row[avg_st_col]) else 0.15
                    except: avg_st = 0.15
                    class_val = parse_class_rank(row[class_col]) if class_col and pd.notna(row[class_col]) else 2.0
                    
                    races_data[rid][boat_i] = {
                        "nat_win": nat_win,
                        "loc_win": loc_win,
                        "mot_2ren": mot_2ren,
                        "avg_st": avg_st,
                        "class_val": class_val
                    }
        except Exception:
            continue
    return races_data

def load_repository_historical_data(repo_root: Path):
    historical_races = []
    od3_root = repo_root / "data" / "previews" / "od3"
    payouts_root = repo_root / "data" / "results" / "payouts"
    
    payouts_dict = {}
    for p_file in payouts_root.glob("**/*.csv"):
        try:
            df = pd.read_csv(p_file)
            for _, row in df.iterrows():
                rid = ""
                for col in ["レースコード", "race_id", "id", "RACE_ID"]:
                    if col in df.columns and pd.notna(row[col]):
                        rid = str(row[col]).strip()
                        break
                if not rid: continue
                for col in ["3連単_組番", "trifecta", "3rentan", "result"]:
                    if col in df.columns and pd.notna(row[col]):
                        payouts_dict[rid] = str(row[col]).strip()
                        break
        except Exception: continue

    races_data = load_race_cards_dataset(repo_root)
    sui_data = load_sui_dataset(repo_root)
    ex_data = load_original_exhibition_dataset(repo_root)

    for od3_csv in od3_root.glob("**/*.csv"):
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
                sui = sui_data.get(rid, {"wave_height": 1.0})
                ex = ex_data.get(rid, {})

                wave = sui["wave_height"]
                rough_factor = 1.0 + (max(0.0, wave - 5.0) * 0.005)

                boat_powers = {}
                for b_i in range(1, 7):
                    if b_i not in boats: continue
                    f = boats[b_i]
                    
                    course_weights = {1: 1.6, 2: 1.1, 3: 1.0, 4: 0.95, 5: 0.85, 6: 0.75}
                    c_bonus = course_weights.get(b_i, 1.0) / rough_factor if b_i == 1 else course_weights.get(b_i, 1.0) * rough_factor

                    ability_score = (f["nat_win"] * 0.4) + (f["loc_win"] * 0.2) + (f["class_val"] * 0.4)
                    ex_time = ex.get(b_i, {}).get("ex_time", 6.8)
                    motor_score = (f["mot_2ren"] / 10.0 * 0.6) + (max(0.0, (7.0 - ex_time) * 10.0) * 0.4)
                    st_score = max(0.0, (0.23 - f["avg_st"]) * 15.0)

                    power = (ability_score * 0.35 + motor_score * 0.35 + st_score * 0.3) * c_bonus
                    boat_powers[b_i] = max(power, 0.1)

                total_power = sum(boat_powers.values())
                boat_win_probs = {b: p / total_power for b, p in boat_powers.items()} if total_power > 0 else {b: 1/6 for b in range(1, 7)}

                raw_odds = {}
                for col in df_od3.columns:
                    if "-" in col:
                        clean_key = col.replace("3連単_", "").replace("3連複_", "").strip()
                        if "-" in clean_key:
                            try:
                                val = float(row[col])
                                if 50.0 <= val <= 300.0:
                                    raw_odds[clean_key] = val
                            except ValueError: pass
                
                if not raw_odds: continue

                probs = {}
                for k, o in raw_odds.items():
                    parts = k.split("-")
                    if len(parts) == 3:
                        try:
                            h1, h2, h3 = int(parts[0]), int(parts[1]), int(parts[2])
                            p1 = boat_win_probs.get(h1, 1/6)
                            p2 = boat_win_probs.get(h2, 1/6) / (1.0 - p1 + 1e-6)
                            p3 = boat_win_probs.get(h3, 1/6) / (1.0 - p1 - p2 + 1e-6)
                            base_p = max(p1 * p2 * p3, 1e-6)
                        except: base_p = 1.0 / o
                    else: base_p = 1.0 / o
                    
                    market_implied_p = 1.0 / o
                    probs[k] = base_p * 0.7 + market_implied_p * 0.3

                prob_sum = sum(probs.values())
                if prob_sum > 0:
                    probs = {k: p_val / prob_sum for k, p_val in probs.items()}

                # 【本格的なレース厳選フィルター】
                # 予測の最大確率が低い（＝展開が読めていない）レースはバッサリ切る（これでレース数を大幅に絞る）
                max_comb_prob = max(probs.values()) if probs else 0
                if max_comb_prob < 0.035:  
                    continue

                valid_bets = []
                for k, o in raw_odds.items():
                    # 50〜200倍のゾーンにボーナスを与え、200倍以上の超大穴への過剰偏重をしっかり抑える
                    if 50.0 <= o < 100.0:
                        odds_multiplier = 1.25  # 50-100倍を拾いやすくする
                    elif 100.0 <= o < 200.0:
                        odds_multiplier = 1.10  # 100-200倍も適度に評価
                    else:
                        odds_multiplier = 0.85  # 200倍以上は少し割引して偏りを防ぐ

                    ev = probs.get(k, 0) * o * odds_multiplier
                    if ev >= 1.35:
                        valid_bets.append((k, o, ev))
                
                if not valid_bets: continue

                # 自信のある上位1点のみに絞り込んで、無駄な点数を減らす
                valid_bets.sort(key=lambda x: x[2], reverse=True)
                top_bets = valid_bets[:1]

                odds_dict = {k: o for k, o, ev in top_bets}
                filtered_probs = {k: probs[k] for k, o, ev in top_bets}

                if not odds_dict: continue

                historical_races.append({
                    "id": rid,
                    "volatility": volatility,
                    "probs": filtered_probs,
                    "odds": odds_dict,
                    "actual_result": str(payouts_dict[rid]).strip()
                })
        except Exception: continue

    print(f"Successfully matched and filtered {len(historical_races)} races for backtest (Balanced & Strictly Filtered Model).")
    return historical_races

def main():
    repo_root = Path(__file__).resolve().parents[3]
    predictor = V12LongshotSkewPredictor()
    
    historical_data = load_repository_historical_data(repo_root)
    if not historical_data:
        print("No historical data could be loaded.")
        return
    
    total_races_bet = len(historical_data)
    total_bets = sum(len(r['odds']) for r in historical_data)
    avg_bets = total_bets / total_races_bet if total_races_bet > 0 else 0
    
    odds_ranges = {"50-100倍": 0, "100-200倍": 0, "200-300倍": 0}
    hit_ranges = {"50-100倍": 0, "100-200倍": 0, "200-300倍": 0}
    hit_count = 0

    for r in historical_data:
        actual = r['actual_result']
        for k, o in r['odds'].items():
            if 50.0 <= o < 100.0: odds_ranges["50-100倍"] += 1
            elif 100.0 <= o < 200.0: odds_ranges["100-200倍"] += 1
            elif 200.0 <= o <= 300.0: odds_ranges["200-300倍"] += 1
            
            if k == actual:
                hit_count += 1
                if 50.0 <= o < 100.0: hit_ranges["50-100倍"] += 1
                elif 100.0 <= o < 200.0: hit_ranges["100-200倍"] += 1
                elif 200.0 <= o <= 300.0: hit_ranges["200-300倍"] += 1

    print("\n=== 【厳選バランス勝負・詳細内訳】 ===")
    print(f"総購入レース数: {total_races_bet:,} レース")
    print(f"総購入点数（延べ）: {total_bets:,} 点")
    print(f"1レースあたりの平均購入点数: {avg_bets:.2f} 点/レース")
    print(f"購入オッズ帯別内訳: {odds_ranges}")
    print(f"的中総数: {hit_count:,} 本")
    print(f"的中オッズ帯別内訳: {hit_ranges}")
    print("----------------------------------------")

    results = predictor.backtest_simulation(historical_data, initial_bankroll=1000000)
    
    print(f"\n=== V23 Balanced & Filtered Backtest Simulation ({len(historical_data)} races) ===")
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

