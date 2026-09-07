import sys
from pathlib import Path
import pandas as pd
import numpy as np
import traceback

root_path = Path(__file__).resolve().parents[3]
sys.path.append(str(root_path))

def parse_class_rank(val):
    s = str(val).strip().upper()
    if "A1" in s: return 4.0
    if "A2" in s: return 3.0
    if "B1" in s: return 2.0
    if "B2" in s: return 1.0
    return 2.0

def get_venue_name_and_code(rid: str, row_data: dict = None, file_path: Path = None):
    venue_codes = {
        "01": "桐生", "02": "戸田", "03": "江戸川", "04": "平和島", "05": "多摩川",
        "06": "浜名湖", "07": "蒲郡", "08": "常滑", "09": "津", "10": "三国",
        "11": "びわこ", "12": "住之江", "13": "尼崎", "14": "鳴門", "15": "丸亀",
        "16": "児島", "17": "宮島", "18": "徳山", "19": "下関", "20": "若松",
        "21": "芦屋", "22": "福岡", "23": "唐津", "24": "大村"
    }
    
    if row_data:
        for col in ["レース場", "venue", "venue_code", "場コード"]:
            if col in row_data and pd.notna(row_data[col]):
                val = str(row_data[col]).strip().zfill(2)
                if val in venue_codes:
                    return val, venue_codes[val]
                for k, v in venue_codes.items():
                    if row_data[col] == v:
                        return k, v

    rid_str = str(rid).strip()
    if len(rid_str) >= 10:
        code_candidate = rid_str[8:10]
        if code_candidate in venue_codes:
            return code_candidate, venue_codes[code_candidate]

    for code, name in venue_codes.items():
        if f"_{code}_" in rid_str or rid_str.startswith(code):
            return code, name
            
    for code, name in venue_codes.items():
        if name in rid_str or (file_path and name in str(file_path)):
            return code, name
            
    return "02", "戸田"

def get_season_by_date(rid: str) -> str:
    rid_str = str(rid).strip()
    if len(rid_str) >= 8:
        try:
            month = int(rid_str[4:6])
            if month in [3, 4, 5]: return "春"
            if month in [6, 7, 8]: return "夏"
            if month in [9, 10, 11]: return "秋"
            return "冬"
        except:
            pass
    return "夏"

def load_stadium_win_rates(repo_root: Path):
    win_rate_path = repo_root / "data" / "estimate" / "stadium" / "win_rate.csv"
    stadium_weights = {}
    if not win_rate_path.exists():
        return stadium_weights
    try:
        df = pd.read_csv(win_rate_path)
        for _, row in df.iterrows():
            v_code = str(row.get("場コード", "")).strip().zfill(2)
            season = str(row.get("季節", "")).strip()
            if not v_code or not season: continue
            
            weights = {}
            for i in range(1, 7):
                col_name = f"{i}コース勝率"
                if col_name in df.columns:
                    try: weights[i] = float(row[col_name])
                    except: weights[i] = 5.0
            stadium_weights[(v_code, season)] = weights
    except Exception:
        pass
    return stadium_weights

def load_sui_dataset(repo_root: Path):
    sui_root = repo_root / "data" / "previews" / "sui"
    sui_data = {}
    if not sui_root.exists(): return sui_data
    for f in sui_root.glob("**/*.csv"):
        try:
            df = pd.read_csv(f)
            for _, row in df.iterrows():
                rid = ""
                for col in ["レースコード", "race_id", "id", "RACE_ID"]:
                    if col in df.columns and pd.notna(row[col]):
                        rid = str(row[col]).strip(); break
                if not rid: continue
                wave_height = 1.0
                for col in ["波の高さ(cm)", "wave_height", "波高"]:
                    if col in df.columns and pd.notna(row[col]):
                        try: wave_height = float(row[col]); break
                        except: pass
                
                wind_speed = 0.0
                for col in ["風速(m)", "wind_speed", "風速"]:
                    if col in df.columns and pd.notna(row[col]):
                        try: wind_speed = float(row[col]); break
                        except: pass

                wind_direction = 0
                for col in ["風向", "wind_direction"]:
                    if col in df.columns and pd.notna(row[col]):
                        try: wind_direction = int(row[col]); break
                        except: pass

                sui_data[rid] = {
                    "wave_height": wave_height,
                    "wind_speed": wind_speed,
                    "wind_direction": wind_direction
                }
        except Exception:
            continue
    return sui_data

def load_stt_dataset(repo_root: Path):
    stt_root = repo_root / "data" / "previews" / "stt"
    stt_data = {}
    if not stt_root.exists(): return stt_data
    for f in stt_root.glob("**/*.csv"):
        try:
            df = pd.read_csv(f)
            for _, row in df.iterrows():
                rid = ""
                for col in ["レースコード", "race_id", "id", "RACE_ID"]:
                    if col in df.columns and pd.notna(row[col]):
                        rid = str(row[col]).strip(); break
                if not rid: continue
                
                boat_courses = {}
                boat_sts = {}
                for boat_i in range(1, 7):
                    c_col = next((c for c in [f"艇{boat_i}_コース", f"boat_{boat_i}_course"] if c in df.columns), None)
                    if c_col and pd.notna(row[c_col]):
                        try: boat_courses[boat_i] = int(row[c_col])
                        except: boat_courses[boat_i] = boat_i
                    else:
                        boat_courses[boat_i] = boat_i

                    st_col = next((c for c in [f"艇{boat_i}_スタート展示", f"boat_{boat_i}_st"] if c in df.columns), None)
                    if st_col and pd.notna(row[st_col]):
                        try: boat_sts[boat_i] = float(row[st_col])
                        except: boat_sts[boat_i] = 0.15
                    else:
                        boat_sts[boat_i] = 0.15

                stt_data[rid] = {"courses": boat_courses, "sts": boat_sts}
        except Exception:
            continue
    return stt_data

def load_original_exhibition_dataset(repo_root: Path):
    ex_root = repo_root / "data" / "previews" / "original_exhibition"
    ex_data = {}
    if not ex_root.exists(): return ex_data
    for f in ex_root.glob("**/*.csv"):
        try:
            df = pd.read_csv(f)
            for _, row in df.iterrows():
                rid = ""
                for col in ["レースコード", "race_id", "id", "RACE_ID"]:
                    if col in df.columns and pd.notna(row[col]):
                        rid = str(row[col]).strip(); break
                if not rid: continue
                if rid not in ex_data: ex_data[rid] = {}
                for boat_i in range(1, 7):
                    prefix = f"艇{boat_i}_"
                    time_val = 6.8
                    for c in [f"{prefix}展示タイム", f"{prefix}タイム", f"boat_{boat_i}_ex_time"]:
                        if c in df.columns and pd.notna(row[c]):
                            try: time_val = float(row[c]); break
                            except: pass
                    
                    turn_val = 37.0
                    for c in [f"{prefix}値1", f"{prefix}まわり足", f"boat_{boat_i}_turn"]:
                        if c in df.columns and pd.notna(row[c]):
                            try: turn_val = float(row[c]); break
                            except: pass

                    straight_val = 5.8
                    for c in [f"{prefix}値2", f"{prefix}直線足", f"boat_{boat_i}_straight"]:
                        if c in df.columns and pd.notna(row[c]):
                            try: straight_val = float(row[c]); break
                            except: pass

                    ex_data[rid][boat_i] = {
                        "ex_time": time_val,
                        "turn_val": turn_val,
                        "straight_val": straight_val
                    }
        except Exception:
            continue
    return ex_data

def load_race_cards_dataset(repo_root: Path):
    cards_root = repo_root / "data" / "programs" / "race_cards"
    races_data = {}
    if not cards_root.exists(): return races_data
    for c_file in cards_root.glob("**/*.csv"):
        try:
            df = pd.read_csv(c_file)
            for _, row in df.iterrows():
                rid = ""
                for col in ["レースコード", "race_id", "id", "RACE_ID"]:
                    if col in df.columns and pd.notna(row[col]):
                        rid = str(row[col]).strip(); break
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
                        "nat_win": nat_win, "loc_win": loc_win, "mot_2ren": mot_2ren,
                        "avg_st": avg_st, "class_val": class_val,
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
                            rid = str(row[col]).strip(); break
                    if not rid: continue
                    for col in ["3連単_組番", "trifecta", "3rentan", "result"]:
                        if col in df.columns and pd.notna(row[col]):
                            payouts_dict[rid] = str(row[col]).strip(); break
            except Exception:
                continue

    races_data = load_race_cards_dataset(repo_root)
    sui_data = load_sui_dataset(repo_root)
    stt_data = load_stt_dataset(repo_root)
    ex_data = load_original_exhibition_dataset(repo_root)
    stadium_win_rates = load_stadium_win_rates(repo_root)

    if not od3_root.exists():
        return historical_races

    for od3_csv in od3_root.glob("**/*.csv"):
        try:
            df_od3 = pd.read_csv(od3_csv)
            for _, row in df_od3.iterrows():
                rid = ""
                for col in ["レースコード", "race_id", "id", "RACE_ID"]:
                    if col in df_od3.columns and pd.notna(row[col]):
                        rid = str(row[col]).strip(); break
                
                if not rid or rid not in payouts_dict or rid not in races_data:
                    continue

                row_dict = row.to_dict()
                v_code, venue = get_venue_name_and_code(rid, row_dict, od3_csv)
                season = get_season_by_date(rid)

                volatility = float(row.get("volatility", 1.5))
                if volatility < 1.2: continue

                boats = races_data[rid]
                sui = sui_data.get(rid, {"wave_height": 1.0, "wind_speed": 0.0, "wind_direction": 0})
                stt = stt_data.get(rid, {"courses": {i:i for i in range(1,7)}, "sts": {i:0.15 for i in range(1,7)}})
                ex = ex_data.get(rid, {})

                wave = sui["wave_height"]
                if wave > 15.0: continue

                wind_speed = sui["wind_speed"]
                rough_factor = 1.0 + (max(0.0, wave - 5.0) * 0.008) + (max(0.0, wind_speed - 3.0) * 0.005)

                default_weights = {1: 7.0, 2: 5.0, 3: 5.0, 4: 4.8, 5: 4.5, 6: 3.0}
                raw_stadium_weights = stadium_win_rates.get((v_code, season), default_weights)
                
                course_weights = {}
                for b_i in range(1, 7):
                    d_w = default_weights.get(b_i, 5.0)
                    s_w = raw_stadium_weights.get(b_i, d_w)
                    course_weights[b_i] = d_w * 0.9 + s_w * 0.1

                boat_powers = {}
                actual_courses = stt["courses"]
                boat_sts = stt["sts"]

                for b_i in range(1, 7):
                    if b_i not in boats: continue
                    f = boats[b_i]
                    
                    real_c = actual_courses.get(b_i, b_i)
                    course_shift_penalty = 1.0
                    if b_i == 1 and real_c > 1:
                        course_shift_penalty = 0.75
                    elif real_c < b_i:
                        course_shift_penalty = 1.15

                    c_base = course_weights.get(b_i, 5.0)
                    c_bonus = ((c_base / 5.0) / (rough_factor ** 1.3) if b_i == 1 else (c_base / 5.0) * (rough_factor ** 1.3)) * course_shift_penalty

                    ability_score = (f["nat_win"] * 0.2) + (f["loc_win"] * 0.1) + (f["class_val"] * 0.3)
                    
                    ex_info = ex.get(b_i, {"ex_time": 6.8, "turn_val": 37.0, "straight_val": 5.8})
                    ex_time = ex_info["ex_time"]
                    turn_val = ex_info["turn_val"]
                    
                    motor_score = (f["mot_2ren"] / 10.0 * 0.4) + (max(0.0, (7.0 - ex_time) * 10.0) * 0.3) + (min(turn_val, 40.0) * 0.3)
                    
                    real_st = boat_sts.get(b_i, f["avg_st"])
                    st_score = max(0.0, (0.23 - real_st) * 20.0)
                    
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
                            try: raw_odds[clean_key] = float(row[col])
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

                if len(target_odds_combos) < 2: continue

                sorted_target = sorted(target_odds_combos.items(), key=lambda x: x[1], reverse=True)
                
                odds_dict = {}
                filtered_probs = {}
                for k, ev in sorted_target[:2]:
                    if k in raw_odds:
                        odds_dict[k] = raw_odds[k]
                        filtered_probs[k] = comb_probs[k]

                if not odds_dict: continue

                historical_races.append({
                    "id": rid, "venue": venue, "volatility": volatility,
                    "probs": filtered_probs, "odds": odds_dict,
                    "actual_result": str(payouts_dict[rid]).strip()
                })
        except Exception:
            continue

    print(f"Successfully matched and filtered {len(historical_races)} races (All Venues + Multi-Data Integration Model).")
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
            
            if current_bankroll <= 0: break
                
            bet_amount = 400
            race_investment = 0
            race_payout = 0
            race_hit = False
            
            for k, o in odds_dict.items():
                actual_bet = bet_amount if current_bankroll >= bet_amount else max(100, int(current_bankroll / 100) * 100)
                if actual_bet <= 0: continue
                
                current_bankroll -= actual_bet
                total_investment += actual_bet
                race_investment += actual_bet
                
                if k == actual:
                    payout = actual_bet * o
                    current_bankroll += payout
                    total_payout += payout
                    race_payout += payout
        
