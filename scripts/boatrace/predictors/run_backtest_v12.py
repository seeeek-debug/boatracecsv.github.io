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

def get_venue_name_and_code(rid, row_data=None, file_path=None):
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
                if val in venue_codes: return val, venue_codes[val]
                for k, v in venue_codes.items():
                    if str(row_data[col]).strip() == v or (k == "11" and "琵琶湖" in str(row_data[col])):
                        return k, v
    rid_str = str(rid).strip()
    if len(rid_str) >= 10:
        c = rid_str[8:10]
        if c in venue_codes: return c, venue_codes[c]
    for code, name in venue_codes.items():
        if f"_{code}_" in rid_str or rid_str.startswith(code): return code, name
        if name in rid_str or (code == "11" and "琵琶湖" in rid_str) or (file_path and (name in str(file_path) or (code == "11" and "琵琶湖" in str(file_path)))):
            return code, name
    return "02", "戸田"

def get_season_by_date(rid):
    rid_str = str(rid).strip()
    if len(rid_str) >= 8:
        try:
            m = int(rid_str[4:6])
            if m in [3, 4, 5]: return "春"
            if m in [6, 7, 8]: return "夏"
            if m in [9, 10, 11]: return "秋"
        except:
            pass
    return "夏"

def load_stadium_win_rates(repo_root: Path):
    win_rate_path = repo_root / "data" / "estimate" / "stadium" / "win_rate.csv"
    stadium_weights = {}
    if not win_rate_path.exists(): return stadium_weights
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
    except:
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
                sui_data[rid] = {"wave_height": wave_height}
        except: continue
    return sui_data

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
                    ex_data[rid][boat_i] = {"ex_time": time_val}
        except: continue
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
        except: continue
    return races_data

# グループA専用関数（浜名湖・芦屋・尼崎・下関：高回収率フルロジック）
def calculate_group_a_logic(boats, sui, ex, stadium_weights, v_code, season):
    wave = sui.get("wave_height", 1.0)
    if wave > 15.0: return None
    rough_factor = 1.0 + (max(0.0, wave - 5.0) * 0.008)

    default_weights = {1: 7.0, 2: 5.0, 3: 5.0, 4: 4.8, 5: 4.5, 6: 3.0}
    raw_stadium_weights = stadium_weights.get((v_code, season), default_weights)
    
    course_weights = {}
    for b_i in range(1, 7):
        d_w = default_weights.get(b_i, 5.0)
        s_w = raw_stadium_weights.get(b_i, d_w)
        course_weights[b_i] = d_w * 0.9 + s_w * 0.1

    boat_powers = {}
    for b_i in range(1, 7):
        if b_i not in boats: continue
        f = boats[b_i]
        c_base = course_weights.get(b_i, 5.0)
        c_bonus = (c_base / 5.0) / (rough_factor ** 1.3) if b_i == 1 else (c_base / 5.0) * (rough_factor ** 1.3)

        ability_score = (f["nat_win"] * 0.2) + (f["loc_win"] * 0.1) + (f["class_val"] * 0.3)
        ex_time = ex.get(b_i, {}).get("ex_time", 6.8)
        motor_score = (f["mot_2ren"] / 10.0 * 0.5) + (max(0.0, (7.0 - ex_time) * 12.0) * 0.5)
        st_score = max(0.0, (0.23 - f["avg_st"]) * 20.0)
        tactic_score = (f["kimarite_rate"] * 1.2) - (f["yarare_rate"] * 0.8)

        power = (ability_score * 0.15 + motor_score * 0.35 + st_score * 0.35 + max(0.1, tactic_score) * 0.15) * c_bonus
        boat_powers[b_i] = max(power, 0.1)
    return boat_powers

# グループB専用関数（戸田・江戸川・蒲郡・津・三国・びわこ：シンプルロジック）
def calculate_group_b_logic(boats):
    boat_powers = {}
    for i in range(1, 7):
        if i not in boats: continue
        f = boats[i]
        power = (f["nat_win"] * 0.2 + f["loc_win"] * 0.1 + f["class_val"] * 0.3 + f["mot_2ren"] * 0.4)
        boat_powers[i] = max(power, 0.1)
    return boat_powers

def main():
    repo_root = Path(__file__).resolve().parents[3]
    od3_root = repo_root / "data" / "previews" / "od3"
    payouts_root = repo_root / "data" / "results" / "payouts"
    
    group_a_venues = ["浜名湖", "芦屋", "尼崎", "下関"]
    group_b_venues = ["戸田", "江戸川", "蒲郡", "津", "三国", "びわこ"]
    allowed_venues = group_a_venues + group_b_venues

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
            except: continue

    races_data = load_race_cards_dataset(repo_root)
    sui_data = load_sui_dataset(repo_root)
    ex_data = load_original_exhibition_dataset(repo_root)
    stadium_win_rates = load_stadium_win_rates(repo_root)

    historical_races = []
    if od3_root.exists():
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
                    if venue not in allowed_venues:
                        continue

                    season = get_season_by_date(rid)
                    volatility = float(row.get("volatility", 1.5))
                    if volatility < 1.2: continue

                    boats = races_data[rid]
                    raw_odds = {}
                    for col in df_od3.columns:
                        if "-" in col:
                            k = col.replace("3連単_", "").replace("3連複_", "").strip()
                            if "-" in k:
                                try: raw_odds[k] = float(row[col])
                                except: pass
                    if not raw_odds: continue

                    if venue in group_a_venues:
                        sui = sui_data.get(rid, {"wave_height": 1.0})
                        ex = ex_data.get(rid, {})
                        boat_powers = calculate_group_a_logic(boats, sui, ex, stadium_win_rates, v_code, season)
                        if boat_powers is None: continue
                    elif venue in group_b_venues:
                        boat_powers = calculate_group_b_logic(boats)
                    else:
                        continue

                    total_p = sum(boat_powers.values())
                    boat_probs = {b: p / total_p for b, p in boat_powers.items()} if total_p > 0 else {b: 1/6 for b in range(1, 7)}

                    comb_probs = {}
                    for h1 in range(1, 7):
                        for h2 in range(1, 7):
                            if h2 == h1: continue
                            for h3 in range(1, 7):
                                if h3 == h1 or h3 == h2: continue
                                k = f"{h1}-{h2}-{h3}"
                                if k in raw_odds:
                                    p1 = boat_probs.get(h1, 1/6)
                                    p2 = boat_probs.get(h2, 1/6) / (1.0 - p1 + 1e-6)
                                    p3 = boat_probs.get(h3, 1/6) / (1.0 - p1 - p2 + 1e-6)
                                    base_p = max(p1 * p2 * p3, 1e-6)
                                    comb_probs[k] = base_p

                    p_sum = sum(comb_probs.values())
                    if p_sum > 0:
                        comb_probs = {k: v / p_sum for k, v in comb_probs.items()}

                    target = {}
                    for k, p in comb_probs.items():
                        if k in raw_odds:
                            o = raw_odds[k]
                            if 30.0 <= o <= 50.0 and p >= 0.02:
                                target[k] = p * o

                    if len(target) < 2: continue
                    sorted_t = sorted(target.items(), key=lambda x: x[1], reverse=True)
                    
                    odds_dict = {}
                    filtered_probs = {}
                    for k, _ in sorted_t[:2]:
                        odds_dict[k] = raw_odds[k]
                        filtered_probs[k] = comb_probs[k]

                    if not odds_dict: continue
                    historical_races.append({
                        "id": rid, "venue": venue, "odds": odds_dict, "actual_result": payouts_dict[rid]
                    })
            except:
                continue

    initial_bankroll = 100000.0
    current_bankroll = initial_bankroll
    total_investment = 0.0
    total_payout = 0.0
    hit_count = 0
    venue_results = {}

    for r in historical_races:
        actual = r['actual_result']
        v = r['venue']
        if v not in venue_results:
            venue_results[v] = {"races": 0, "hits": 0, "investment": 0.0, "payout": 0.0}
        venue_results[v]["races"] += 1
        
        race_hit = False
        for k, o in r['odds'].items():
            bet = 400 if current_bankroll >= 400 else 100
            current_bankroll -= bet
            total_investment += bet
            venue_results[v]["investment"] += bet
            if k == actual:
                payout = bet * o
                current_bankroll += payout
                total_payout += payout
                venue_results[v]["payout"] += payout
                hit_count += 1
                race_hit = True
        if race_hit: venue_results[v]["hits"] += 1

    roi = (total_payout / total_investment * 100) if total_investment > 0 else 0.0
    print(f"\n=== 【完全分離デュアルロジック混成モデル結果】 ===")
    print(f"総購入レース数: {len(historical_races):,} レース")
    print(f"的中総数: {hit_count:,} 本")
    print(f"初期資金: ¥{int(initial_bankroll):,}")
    print(f"最終資金: ¥{current_bankroll:,.2f}")
    print(f"総投資額: ¥{total_investment:,.2f}")
    print(f"総払戻金: ¥{total_payout:,.2f}")
    print(f"回収率 (ROI): {roi:.2f}%")
    
    print("\n=== 【開催場別の成績詳細】 ===")
    for v, stats in venue_results.items():
        v_inv = int(stats['investment'])
        v_pay = int(stats['payout'])
        v_roi = (v_pay / v_inv * 100) if v_inv > 0 else 0.0
        print(f"{v:4s} | 購入R: {stats['races']:3d} | 的中: {stats['hits']:2d} | 投資: ¥{v_inv:6,d} | 払戻: ¥{v_pay:6,d} | 回収率: {v_roi:5.1f}%")

if __name__ == "__main__":
    main()

