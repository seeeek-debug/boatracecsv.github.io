import sys
from pathlib import Path
import pandas as pd
import numpy as np
import traceback

root_path = Path(__file__).resolve().parents[3]
sys.path.append(str(root_path))

def parse_class_rank(val):
    """選手級別を数値化する"""
    s = str(val).strip().upper()
    if "A1" in s: return 4.0
    if "A2" in s: return 3.0
    if "B1" in s: return 2.0
    if "B2" in s: return 1.0
    return 2.0

def get_venue_name(rid: str, file_path: Path = None) -> str:
    """レースIDやファイルパスから開催場名を特定する"""
    venue_names = ["桐生", "戸田", "江戸川", "平和島", "多摩川", "浜名湖", "蒲郡", "常滑", "津", "三国", "びわこ", "住之江", "尼崎", "鳴門", "丸亀", "児島", "宮島", "徳山", "下関", "若松", "芦屋", "福岡", "唐津", "大村"]
    for v in venue_names:
        if v in rid or (file_path and v in str(file_path)):
            return v
    venue_codes = {
        "01": "桐生", "02": "戸田", "03": "江戸川", "04": "平和島", "05": "多摩川",
        "06": "浜名湖", "07": "蒲郡", "08": "常滑", "09": "津", "10": "三国",
        "11": "びわこ", "12": "住之江", "13": "尼崎", "14": "鳴門", "15": "丸亀",
        "16": "児島", "17": "宮島", "18": "徳山", "19": "下関", "20": "若松",
        "21": "芦屋", "22": "福岡", "23": "唐津", "24": "大村"
    }
    for code, name in venue_codes.items():
        if code in rid or (file_path and f"_{code}_" in str(file_path)):
            return name
    return "その他"

def load_sui_dataset(repo_root: Path):
    """水面コンディション（波高）をロードする"""
    sui_root = repo_root / "data" / "previews" / "sui"
    sui_data = {}
    if not sui_root.exists():
        return sui_data
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
    if not ex_root.exists():
        return ex_data
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
    if not cards_root.exists():
        return races_data
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
                        "class_val": class_val,
                        "kimarite_rate": 0.7 if boat_i == 1 else (0.4 if boat_i in [2, 3, 4] else 0.2),
                        "yarare_rate": 0.3 if boat_i == 1 else (0.6 if boat_i in [2, 3, 4] else 0.8)
                    }
        except Exception:
            continue
    return races_data

def load_repository_historical_data(repo_root: Path):
    historical_races = []
    od3_root = repo_root / "data" / "previews" / "od3"
    payouts_root = repo_root / "data" / "results" / "payouts"
    
    payouts_dict = {}
    if payouts_root.exists():
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

    if not od3_root.exists():
        print(f"Warning: {od3_root} does not exist.")
        return historical_races

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

                venue = get_venue_name(rid, od3_csv)

                volatility = float(row.get("volatility", 1.5))
                if volatility < 1.2:
                    continue

                boats = races_data[rid]
                sui = sui_data.get(rid, {"wave_height": 1.0})
                ex = ex_data.get(rid, {})

                wave = sui["wave_height"]
                if wave > 15.0:
                    continue

                rough_factor = 1.0 + (max(0.0, wave - 5.0) * 0.008)

                boat_powers = {}
                for b_i in range(1, 7):
                    if b_i not in boats: continue
                    f = boats[b_i]
                    
                    course_weights = {1: 1.75, 2: 1.12, 3: 1.02, 4: 0.92, 5: 0.82, 6: 0.72}
                    c_bonus = course_weights.get(b_i, 1.0) / (rough_factor ** 1.3) if b_i == 1 else course_weights.get(b_i, 1.0) * (rough_factor ** 1.3)

                    ability_score = (f["nat_win"] * 0.2) + (f["loc_win"] * 0.1) + (f["class_val"] * 0.3)
                    ex_time = ex.get(b_i, {}).get("ex_time", 6.8)
                    motor_score = (f["mot_2ren"] / 10.0 * 0.5) + (max(0.0, (7.0 - ex_time) * 12.0) * 0.5)
                    st_score = max(0.0, (0.23 - f["avg_st"]) * 20.0)
                    tactic_score = (f["kimarite_rate"] * 1.2) - (f["yarare_rate"] * 0.8)

                    power = (ability_score * 0.15 + motor_score * 0.35 + st_score * 0.35 + max(0.1, tactic_score) * 0.15) * c_bonus
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
                                raw_odds[clean_key] = val
                            except ValueError: pass
                
                if not raw_odds: continue

                comb_probs = {}
                for h1 in range(1, 7):
                    for h2 in range(1, 7):
                        if h2 == h1: continue
                        for h3 in range(1, 7):
                            if h3 == h1 or h3 == h2: continue
                            k = f"{h1}-{h2}-{h3}"
                            if k in raw_odds:
                                p1 = boat_win_probs.get(h1, 1/6)
                                p2 = boat_win_probs.get(h2, 1/6) / (1.0 - p1 + 1e-6)
                                p3 = boat_win_probs.get(h3, 1/6) / (1.0 - p1 - p2 + 1e-6)
                                base_p = max(p1 * p2 * p3, 1e-6)

                                kimarite_scenario_bias = 1.0
                                if h1 == 1:
                                    h2_pwr = boat_powers.get(h2, 1.0)
                                    h3_pwr = boat_powers.get(h3, 1.0)
                                    kimarite_scenario_bias = 1.0 + (h2_pwr + h3_pwr) * 0.05
                                elif h1 in [3, 4]:
                                    kimarite_scenario_bias = 1.3
                                
                                base_p *= kimarite_scenario_bias
                                comb_probs[k] = base_p

                prob_sum = sum(comb_probs.values())
                if prob_sum > 0:
                    comb_probs = {k: p_val / prob_sum for k, p_val in comb_probs.items()}

                if not comb_probs: continue

                target_odds_combos = {}
                for k, p in comb_probs.items():
                    if k in raw_odds:
                        odds_val = raw_odds[k]
                        if 30.0 <= odds_val <= 50.0 and p >= 0.02:
                            target_odds_combos[k] = p * odds_val

                if len(target_odds_combos) < 2:
                    continue

                sorted_target = sorted(target_odds_combos.items(), key=lambda x: x[1], reverse=True)
                
                odds_dict = {}
                filtered_probs = {}
                for k, ev in sorted_target[:2]:
                    if k in raw_odds:
                        odds_dict[k] = raw_odds[k]
                        filtered_probs[k] = comb_probs[k]

                if not odds_dict: continue

                historical_races.append({
                    "id": rid,
                    "venue": venue,
                    "volatility": volatility,
                    "probs": filtered_probs,
                    "odds": odds_dict,
                    "actual_result": str(payouts_dict[rid]).strip()
                })
        except Exception: continue

    print(f"Successfully matched and filtered {len(historical_races)} races (All-Venue 30-50x Model).")
    return historical_races

def main():
    try:
        repo_root = Path(__file__).resolve().parents[3]
        
        historical_data = load_repository_historical_data(repo_root)
        if not historical_data:
            print("No historical data could be loaded.")
            return
        
        total_races_bet = len(historical_data)
        total_bets = sum(len(r['odds']) for r in historical_data)
        
        hit_count = 0
        venue_results = {}

        initial_bankroll = 100000.0
        current_bankroll = initial_bankroll
        total_investment = 0.0
        total_payout = 0.0
        max_bankroll = initial_bankroll
        max_drawdown = 0.0
        
        for r in historical_data:
            actual = r['actual_result']
            odds_dict = r['odds']
            v = r['venue']
            
            if v not in venue_results:
                venue_results[v] = {"races": 0, "hits": 0, "investment": 0.0, "payout": 0.0}
            venue_results[v]["races"] += 1
            
            if current_bankroll <= 0:
                break
                
            raw_bet = current_bankroll * 0.001
            bet_amount = max(100, min(500, int(raw_bet / 100) * 100))
            
            race_investment = 0
            race_payout = 0
            race_hit = False
            
            for k, o in odds_dict.items():
                if current_bankroll < bet_amount:
                    actual_bet = max(100, int(current_bankroll / 100) * 100)
                    if actual_bet < 100: actual_bet = 0
                else:
                    actual_bet = bet_amount
                    
                if actual_bet <= 0: continue
                
                current_bankroll -= actual_bet
                total_investment += actual_bet
                race_investment += actual_bet
                
                if k == actual:
                    payout = actual_bet * o
                    current_bankroll += payout
                    total_payout += payout
                    race_payout += payout
                    hit_count += 1
                    race_hit = True
            
            if race_hit:
                venue_results[v]["hits"] += 1
                
            venue_results[v]["investment"] += race_investment
            venue_results[v]["payout"] += race_payout
                    
            if current_bankroll > max_bankroll:
                max_bankroll = current_bankroll
            drawdown = max_bankroll - current_bankroll
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        roi = (total_payout / total_investment * 100) if total_investment > 0 else 0.0
        max_drawdown_rate = (max_drawdown / max_bankroll * 100) if max_bankroll > 0 else 0.0

        print(f"\n=== 【全場対象 30〜50倍中穴特化・2点買い実績】 ===")
        print(f"総購入レース数: {total_races_bet:,} レース")
        print(f"総購入点数（延べ）: {total_bets:,} 点")
        print(f"的中総数: {hit_count:,} 本")
        print("----------------------------------------")
        print(f"初期資金: ¥{int(initial_bankroll):,}")
        print(f"最終資金: ¥{current_bankroll:,.2f}")
        print(f"総投資額: ¥{total_investment:,.2f}")
        print(f"総払戻金: ¥{total_payout:,.2f}")
        print(f"回収率 (ROI): {roi:.2f}%")
        print(f"最大ドローダウン (金額): ¥{max_drawdown:,.2f}")
        print(f"最大ドローダウン (率): {max_drawdown_rate:.2f}%")
        
        print("\n=== 【開催場別の成績詳細】 ===")
        print(f"{'場名':4s} | {'購入R':5s} | {'的中数':5s} | {'的中率':6s} | {'投資額':10s} | {'払戻金':10s} | {'回収率':6s}")
        print("-" * 65)
        
        sorted_venues = sorted(venue_results.items(), key=lambda x: (x[1]["payout"] / max(1.0, x[1]["investment"])), reverse=True)
        for v, stats in sorted_venues:
            v_races = stats['races']
            v_hits = stats['hits']
            v_hit_rate = (v_hits / v_races * 100) if v_races > 0 else 0.0
            # 整数にキャストしてフォーマットエラーを防ぐ
            v_inv = int(stats['investment'])
            v_pay = int(stats['payout'])
            v_roi = (v_pay / v_inv * 100) if v_inv > 0 else 0.0
            print(f"{v:4s} | {v_races:5d} | {v_hits:5d} | {v_hit_rate:5.1f}% | ¥{v_inv:9,d} | ¥{v_pay:9,d} | {v_roi:5.1f}%")

        print("=== Backtest Finished Successfully ===")
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

