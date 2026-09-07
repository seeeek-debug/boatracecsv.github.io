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
                    if row_data[col] == v: return k, v
    rid_str = str(rid).strip()
    if len(rid_str) >= 10:
        c = rid_str[8:10]
        if c in venue_codes: return c, venue_codes[c]
    for code, name in venue_codes.items():
        if f"_{code}_" in rid_str or rid_str.startswith(code): return code, name
        if name in rid_str or (file_path and name in str(file_path)): return code, name
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

def main():
    repo_root = Path(__file__).resolve().parents[3]
    od3_root = repo_root / "data" / "previews" / "od3"
    payouts_root = repo_root / "data" / "results" / "payouts"
    
    target_venues = ["戸田", "江戸川", "蒲郡", "津", "三国", "びわこ", "浜名湖", "芦屋", "尼崎", "下関"]

    payouts_dict = {}
    if payouts_root.exists():
        for p_file in payouts_root.glob("**/*.csv"):
            df = pd.read_csv(p_file)
            for _, row in df.iterrows():
                for col in ["レースコード", "race_id", "id", "RACE_ID"]:
                    if col in df.columns and pd.notna(row[col]):
                        rid = str(row[col]).strip()
                        for p_col in ["3連単_組番", "trifecta", "3rentan", "result"]:
                            if p_col in df.columns and pd.notna(row[p_col]):
                                payouts_dict[rid] = str(row[p_col]).strip()
                                break
                        break

    races_data = {}
    cards_root = repo_root / "data" / "programs" / "race_cards"
    if cards_root.exists():
        for c_file in cards_root.glob("**/*.csv"):
            df = pd.read_csv(c_file)
            for _, row in df.iterrows():
                rid = ""
                for col in ["レースコード", "race_id", "id", "RACE_ID"]:
                    if col in df.columns and pd.notna(row[col]):
                        rid = str(row[col]).strip()
                        break
                if not rid: continue
                if rid not in races_data: races_data[rid] = {}
                for i in range(1, 7):
                    p = f"艇{i}_"
                    races_data[rid][i] = {
                        "nat_win": float(row.get(f"{p}全国勝率", 5.0) or 5.0),
                        "loc_win": float(row.get(f"{p}當地勝率", 5.0) or 5.0),
                        "mot_2ren": float(row.get(f"{p}モーター2連対率", 30.0) or 30.0),
                        "avg_st": float(row.get(f"{p}平均ST", 0.15) or 0.15),
                        "class_val": parse_class_rank(row.get(f"{p}級別", "B1")),
                        "kimarite_rate": 0.7 if i == 1 else (0.4 if i in [2, 3, 4] else 0.2),
                        "yarare_rate": 0.3 if i == 1 else (0.6 if i in [2, 3, 4] else 0.8)
                    }

    historical_races = []
    if od3_root.exists():
        for od3_csv in od3_root.glob("**/*.csv"):
            df_od3 = pd.read_csv(od3_csv)
            for _, row in df_od3.iterrows():
                rid = ""
                for col in ["レースコード", "race_id", "id", "RACE_ID"]:
                    if col in df_od3.columns and pd.notna(row[col]):
                        rid = str(row[col]).strip()
                        break
                if not rid or rid not in payouts_dict or rid not in races_data:
                    continue
                
                v_code, venue = get_venue_name_and_code(rid, row.to_dict(), od3_csv)
                
                if venue not in target_venues:
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

                boat_powers = {}
                for i in range(1, 7):
                    if i not in boats: continue
                    f = boats[i]
                    power = (f["nat_win"] * 0.2 + f["loc_win"] * 0.1 + f["class_val"] * 0.3 + f["mot_2ren"] * 0.4)
                    boat_powers[i] = max(power, 0.1)

                total_p = sum(boat_powers.values())
                boat_probs = {b: p / total_p for b, p in boat_powers.items()}

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
                                comb_probs[k] = max(p1 * p2 * p3, 1e-6)

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
    print(f"\n=== 【黄金ロジック＋指定10場限定モデル結果】 ===")
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

