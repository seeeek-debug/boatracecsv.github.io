import os
import pandas as pd
import numpy as np

def load_motor_abilities(file_path="data/estimate/motor_ability_score_v4.csv"):
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    return pd.DataFrame()

def load_race_cards(file_path, motor_df):
    if not os.path.exists(file_path):
        return {}
    
    df = pd.read_csv(file_path)
    parsed_races = {}
    
    for idx, row in df.iterrows():
        race_code = None
        for col in ['レースコード', 'race_id', 'race_code', 'R_code']:
            if col in df.columns:
                race_code = str(row.get(col))
                break
        if not race_code:
            race_code = str(row.get('race_id', idx + 1))

        boat_data_list = []
        for boat_num in range(1, 7):
            prefix = f"艇{boat_num}_"
            
            player_id = row.get(f"{prefix}登番", 0)
            player_name = row.get(f"{prefix}選手名", "")
            class_type = row.get(f"{prefix}級別", "B2")
            f_count = row.get(f"{prefix}F本数", 0)
            avg_st = row.get(f"{prefix}全国平均ST", 0.20)
            motor_number = row.get(f"{prefix}モーター番号", 0)
            
            motor_power = 1.0
            if not motor_df.empty and 'motor_number' in motor_df.columns:
                matched_motor = motor_df[motor_df['motor_number'] == motor_number]
                if not matched_motor.empty:
                    motor_power = float(matched_motor.iloc[0].get('ability_score', 1.0))
            
            class_bonus = {"A1": 3.0, "A2": 1.8, "B1": 0.8, "B2": 0.0}.get(class_type, 0.0)
            f_penalty = f_count * 1.5  
            
            boat_data_list.append({
                'boat_number': boat_num,
                'player_id': player_id,
                'player_name': player_name,
                'class_bonus': class_bonus,
                'avg_st': float(avg_st),
                'f_penalty': f_penalty,
                'motor_number': motor_number,
                'motor_power': motor_power
            })
            
        parsed_races[race_code] = boat_data_list
        
    return parsed_races

def load_preview_odds(file_path):
    if not os.path.exists(file_path):
        return {}
    
    df = pd.read_csv(file_path)
    odds_dict = {}
    
    for idx, row in df.iterrows():
        race_code = None
        for col in ['レースコード', 'race_id', 'race_code']:
            if col in df.columns:
                race_code = str(row.get(col)).strip()
                break
        if not race_code:
            continue
            
        race_odds = {}
        for col in df.columns:
            if '3連単' in col:
                raw_part = col.replace('3連単_', '').replace('3連単', '')
                combo = raw_part.replace('=', '-').replace('・', '-')
                if len(combo) == 3 and combo.isdigit():
                    combo = f"{combo[0]}-{combo[1]}-{combo[2]}"
                try:
                    val = float(row[col])
                    race_odds[combo] = val
                except ValueError:
                    pass
        odds_dict[race_code] = race_odds
        
    return odds_dict

def load_stt_data(file_path):
    if not os.path.exists(file_path):
        return {}
    df = pd.read_csv(file_path)
    stt_dict = {}
    for idx, row in df.iterrows():
        race_code = None
        for col in ['レースコード', 'race_id', 'race_code']:
            if col in df.columns:
                race_code = str(row.get(col)).strip()
                break
        if not race_code:
            continue
        
        boat_stt = {}
        for b in range(1, 7):
            course_col = f"艇{b}_コース"
            st_col = f"艇{b}_スタート展示"
            course = int(row.get(course_col, b)) if pd.notna(row.get(course_col)) else b
            st = float(row.get(st_col, 0.15)) if pd.notna(row.get(st_col)) else 0.15
            boat_stt[b] = {'course': course, 'st': st}
        stt_dict[race_code] = boat_stt
    return stt_dict

def load_original_exhibition_data(file_path):
    if not os.path.exists(file_path):
        return {}
    df = pd.read_csv(file_path)
    orig_dict = {}
    for idx, row in df.iterrows():
        race_code = None
        for col in ['レースコード', 'race_id', 'race_code']:
            if col in df.columns:
                race_code = str(row.get(col)).strip()
                break
        if not race_code:
            continue
        
        boat_orig = {}
        for b in range(1, 7):
            v1 = float(row.get(f"艇{b}_値1", 0)) if pd.notna(row.get(f"艇{b}_値1", 0)) else 0.0
            v2 = float(row.get(f"艇{b}_値2", 0)) if pd.notna(row.get(f"艇{b}_値2", 0)) else 0.0
            boat_orig[b] = {'val1': v1, 'val2': v2}
        orig_dict[race_code] = boat_orig
    return orig_dict

def calculate_combination_probabilities(boat_data_list, stt_info, orig_info, stadium_id):
    base_frame_win_bias = {1: 0.60, 2: 0.15, 3: 0.11, 4: 0.08, 5: 0.04, 6: 0.02}
    
    boat_scores = {}
    for data in boat_data_list:
        boat = data['boat_number']
        
        ex_st = 0.15
        if stt_info and boat in stt_info:
            ex_st = stt_info[boat]['st']
        st_score = max(0.25 - ex_st, 0) * 35 if ex_st > 0 else -15.0 
        
        orig_bonus = 0.0
        if orig_info and boat in orig_info:
            val = orig_info[boat]['val1']
            if 35.0 <= val <= 40.0:
                orig_bonus = (40.0 - val) * 0.6
        
        stat_score = data['class_bonus'] + st_score - data['f_penalty'] + orig_bonus
        motor_score = data['motor_power'] * 3.0
        
        raw_power = (base_frame_win_bias[boat] * 15) + stat_score + motor_score
        boat_scores[boat] = max(raw_power, 0.1)
        
    total_score = sum(boat_scores.values())
    base_probs = {boat: score / total_score for boat, score in boat_scores.items()}
    
    combo_probs = {}
    for b1 in range(1, 7):
        p1 = base_probs[b1]
        for b2 in range(1, 7):
            if b2 == b1: continue
            remaining_after_p1 = 1.0 - p1
            p2_conditional = base_probs[b2] / remaining_after_p1 if remaining_after_p1 > 0 else 0.2
            
            for b3 in range(1, 7):
                if b3 == b1 or b3 == b2: continue
                remaining_after_p2 = remaining_after_p1 - base_probs[b2]
                if remaining_after_p2 <= 0:
                    p3_conditional = 1.0 / 4.0
                else:
                    p3_conditional = base_probs[b3] / remaining_after_p2
                    
                combo = f"{b1}-{b2}-{b3}"
                combo_probs[combo] = p1 * p2_conditional * p3_conditional
                
    total_cp = sum(combo_probs.values())
    if total_cp > 0:
        combo_probs = {k: v / total_cp for k, v in combo_probs.items()}
        
    return combo_probs

def generate_target_return_bets(boat_data_list, race_actual_odds, stt_info, orig_info, stadium_id):
    combo_probs = calculate_combination_probabilities(boat_data_list, stt_info, orig_info, stadium_id)
    
    if not race_actual_odds:
        return None, "見送り"

    min_odds = min(race_actual_odds.values()) if race_actual_odds else 10.0
    if min_odds < 8.0:
        return None, "見送り"

    if min_odds < 30.0:
        race_type = "中穴"
        target_payout = 5000
        max_inv = 1000
    else:
        race_type = "穴"
        target_payout = 8000
        max_inv = 800

    valid_bets = []
    for combo, combo_prob in combo_probs.items():
        if combo not in race_actual_odds:
            continue
            
        # ★【超重要】確率が 1.5% 未満の「カス確率の大穴」は、オッズが高くても絶対に対象外にする
        if combo_prob < 0.015:
            continue
            
        actual_odds = race_actual_odds[combo]
        expected_value = combo_prob * actual_odds
        
        if actual_odds < 15.0:
            continue
            
        if expected_value >= 1.20:
            valid_bets.append((combo, combo_prob, actual_odds, expected_value))
                
    if not valid_bets:
        return None, "見送り"
        
    valid_bets.sort(key=lambda x: x[3], reverse=True)
    selected_candidates = valid_bets[:2]
    
    allocated_bets = []
    total_inv = 0
    
    for combo, prob, odds, ev in selected_candidates:
        if odds <= 0: continue
        raw_w = target_payout / odds
        w = max(100, round(raw_w / 100) * 100)
        
        if total_inv + w <= max_inv:
            allocated_bets.append((combo, w, odds))
            total_inv += w
        else:
            if not allocated_bets and max_inv >= 100:
                allocated_bets.append((combo, 100, odds))
                total_inv += 100
            break
            
    if not allocated_bets:
        return None, "見送り"
        
    return allocated_bets, race_type

def run_monthly_backtest(start_date="2026-08-01", end_date="2026-08-31"):
    motor_df = load_motor_abilities()
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    
    total_investment = 0
    total_payout = 0
    hit_count = 0
    total_races = 0
    skipped_races = 0
    
    type_stats = {"中穴": {"count": 0, "hits": 0, "inv": 0, "pay": 0},
                  "穴": {"count": 0, "hits": 0, "inv": 0, "pay": 0}}
    
    print("=== 2026年 8月度 月間一括テスト（確率フィルター導入・厳選版） ===")
    
    for single_date in dates:
        year = single_date.strftime("%Y")
        month = single_date.strftime("%m")
        day = single_date.strftime("%d")
        
        result_path = f"data/results/payouts/{year}/{month}/{day}.csv"
        race_card_path = f"data/programs/race_cards/{year}/{month}/{day}.csv"
        preview_odds_path = f"data/previews/od3/{year}/{month}/{day}.csv"
        stt_path = f"data/previews/stt/{year}/{month}/{day}.csv"
        orig_path = f"data/previews/original_exhibition/{year}/{month}/{day}.csv"
        
        if not os.path.exists(result_path) or not os.path.exists(race_card_path):
            continue
            
        results_df = pd.read_csv(result_path)
        races_dict = load_race_cards(race_card_path, motor_df)
        odds_dict = load_preview_odds(preview_odds_path)
        stt_dict = load_stt_data(stt_path)
        orig_dict = load_original_exhibition_data(orig_path)
        
        for idx, row in results_df.iterrows():
            race_code = ""
            for col in results_df.columns:
                if 'レースコード' in str(col) or 'race_code' in str(col).lower() or 'race_id' in str(col).lower():
                    race_code = str(row.get(col, '')).strip()
                    break
            if not race_code:
                race_code = str(row.get('レースコード', idx + 1))
                
            stadium_id = 12
            for col in results_df.columns:
                if 'レース場' in str(col) or 'stadium' in str(col).lower():
                    try:
                        stadium_id = int(row.get(col, 12))
                    except ValueError:
                        pass
                    break
            
            winning_combo = ""
            if '3連単_組番' in row and pd.notna(row['3連単_組番']):
                raw_combo = str(row['3連単_組番']).strip()
                winning_combo = raw_combo.replace('=', '-')
            
            payout_yen = 0.0
            if '3連単_払戻金' in row and pd.notna(row['3連単_払戻金']):
                try:
                    payout_yen = float(row['3連単_払戻金'])
                except ValueError:
                    payout_yen = 0.0
            
            if race_code not in races_dict:
                continue
                
            boats = races_dict[race_code]
            race_actual_odds = odds_dict.get(race_code, {})
            stt_info = stt_dict.get(race_code, {})
            orig_info = orig_dict.get(race_code, {})
            
            allocated_bets, race_type = generate_target_return_bets(boats, race_actual_odds, stt_info, orig_info, stadium_id)
            
            if allocated_bets is None:
                skipped_races += 1
                continue
            
            investment = sum(amount for combo, amount, odds in allocated_bets)
            total_investment += investment
            total_races += 1
            
            type_stats[race_type]["count"] += 1
            type_stats[race_type]["inv"] += investment
            
            hit = False
            for combo, amount, odds in allocated_bets:
                if combo == winning_combo:
                    hit = True
                    payout_added = (payout_yen / 100) * amount
                    total_payout += payout_added
                    type_stats[race_type]["pay"] += payout_added
                    break
            
            if hit:
                hit_count += 1
                type_stats[race_type]["hits"] += 1

    roi = (total_payout / total_investment * 100) if total_investment > 0 else 0
    hit_rate = (hit_count / total_races * 100) if total_races > 0 else 0
    net_profit = total_payout - total_investment
    
    print("\n" + "="*50)
    print(f" 🎯 2026年 8月度 月間一括テスト最終結果（確率フィルター導入版）")
    print("="*50)
    for r_type, st in type_stats.items():
        t_roi = (st["pay"] / st["inv"] * 100) if st["inv"] > 0 else 0
        t_hit = (st["hits"] / st["count"] * 100) if st["count"] > 0 else 0
        print(f"■ 【{r_type}】 レース数: {st['count']} | 的中数: {st['hits']} ({t_hit:.1f}%) | 投資: {st['inv']:,}円 | 払戻: {st['pay']:,}円 | 回収率: {t_roi:.1f}%")
    print("-" * 50)
    print(f" 見送りレース数 : {skipped_races} レース")
    print(f" 購入レース数   : {total_races} レース")
    print(f" 的中数         : {hit_count} レース (的中率: {hit_rate:.2f}%)")
    print(f" 総投資額       : {total_investment:,.0f} 円")
    print(f" 総払戻金       : {total_payout:,.0f} 円")
    print(f" 収支           : {net_profit:+,.0f} 円")
    print(f" 総合回収率 (ROI) : {roi:.2f}%")
    print("="*50)

if __name__ == "__main__":
    run_monthly_backtest("2026-08-01", "2026-08-31")

