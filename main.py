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
            
            class_bonus = {"A1": 2.5, "A2": 1.5, "B1": 0.6, "B2": 0.0}.get(class_type, 0.0)
            f_penalty = f_count * 1.0  
            
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

def judge_race_condition(wave_height, wind_speed):
    if wave_height <= 5 and wind_speed <= 3:
        return "solid"
    elif wave_height <= 10 and wind_speed <= 5:
        return "medium"
    else:
        return "rough"

def calculate_boat_scores(boat_data_list, condition_type, stadium_id):
    scores = {}
    
    high_in_stadiums = [17, 20, 23]
    low_in_stadiums = [1, 2, 3, 4, 5, 6, 11, 14]
    
    if stadium_id in high_in_stadiums:
        in_bias_multiplier = 1.3
    elif stadium_id in low_in_stadiums:
        in_bias_multiplier = 0.7
    else:
        in_bias_multiplier = 1.0

    if condition_type == "solid":
        w_frame, w_stat, w_motor = 1.2 * in_bias_multiplier, 1.8, 2.0
    elif condition_type == "medium":
        w_frame, w_stat, w_motor = 0.8 * in_bias_multiplier, 1.8, 2.5
    else:
        w_frame, w_stat, w_motor = 0.3 * in_bias_multiplier, 1.5, 3.0

    for data in boat_data_list:
        boat = data['boat_number']
        extra_in_bonus = 1.5 if (boat == 1 and stadium_id in high_in_stadiums) else 1.0
        
        frame_bias = ((7 - boat) * 0.8 * extra_in_bonus) if condition_type != "rough" else (boat if boat >= 4 else 3)
        st_score = max(0.25 - data['avg_st'], 0) * 20
        stat_score = data['class_bonus'] + st_score - data['f_penalty']
        motor_score = data['motor_power'] * w_motor
        
        total_score = (frame_bias * w_frame) + (stat_score * w_stat) + motor_score
        scores[boat] = max(total_score, 0.1)
        
    return scores

def generate_target_return_bets(scores, race_actual_odds):
    total_score = sum(scores.values())
    probs = {boat: score / total_score for boat, score in scores.items()}
    
    bets = []
    for b1 in range(1, 7):
        for b2 in range(1, 7):
            if b2 == b1: continue
            for b3 in range(1, 7):
                if b3 == b1 or b3 == b2: continue
                combo = f"{b1}-{b2}-{b3}"
                p1 = probs[b1]
                p2 = probs[b2] / (1 - p1) if (1 - p1) > 0 else 0
                p3 = probs[b3] / (1 - p1 - probs[b2]) if (1 - p1 - probs[b2]) > 0 else 0
                combo_prob = p1 * p2 * p3
                
                if race_actual_odds and combo in race_actual_odds:
                    actual_odds = race_actual_odds[combo]
                else:
                    actual_odds = 0.75 / combo_prob if combo_prob > 0 else 100.0
                    
                bets.append((combo, combo_prob, actual_odds))
                
    bets.sort(key=lambda x: x[1], reverse=True)
    
    if len(bets) < 3:
        bets = [("1-2-3", 0.2, 10.0), ("1-2-4", 0.15, 15.0), ("1-3-2", 0.1, 20.0)]
        
    top_combo, top_prob, top_odds = bets[0]

    # --- 見送りを廃止し、必ずいずれかのレースタイプに振り分けて購入 ---
    if top_odds < 20.0:
        race_type = "固め"
        min_odds, max_odds = 1.0, 50.0
        target_payout = 4000
        max_inv = 1000
    elif top_odds < 60.0:
        race_type = "中穴"
        min_odds, max_odds = 10.0, 120.0
        target_payout = 8000
        max_inv = 1200
    else:
        race_type = "穴"
        min_odds, max_odds = 20.0, 500.0
        target_payout = 15000
        max_inv = 1200

    target_bets = [b for b in bets if min_odds <= b[2] < max_odds]
    if len(target_bets) < 2:
        target_bets = bets[:5]
        
    selected_candidates = target_bets[:5]
    
    allocated_bets = []
    total_inv = 0
    
    for combo, prob, odds in selected_candidates:
        if odds <= 0: odds = 10.0
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
        allocated_bets = [(bets[0][0], 100, bets[0][2])]
        
    return allocated_bets, race_type

def run_monthly_backtest(start_date="2026-08-01", end_date="2026-08-31"):
    motor_df = load_motor_abilities()
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    
    total_investment = 0
    total_payout = 0
    hit_count = 0
    total_races = 0
    skipped_races = 0
    
    type_stats = {"固め": {"count": 0, "hits": 0, "inv": 0, "pay": 0},
                  "中穴": {"count": 0, "hits": 0, "inv": 0, "pay": 0},
                  "穴": {"count": 0, "hits": 0, "inv": 0, "pay": 0}}
    
    print("=== 2026年 8月度 月間一括バックテスト実行中（全レース完全購入版） ===")
    
    for single_date in dates:
        year = single_date.strftime("%Y")
        month = single_date.strftime("%m")
        day = single_date.strftime("%d")
        
        result_path = f"data/results/payouts/{year}/{month}/{day}.csv"
        race_card_path = f"data/programs/race_cards/{year}/{month}/{day}.csv"
        preview_odds_path = f"data/previews/od3/{year}/{month}/{day}.csv"
        
        if not os.path.exists(result_path) or not os.path.exists(race_card_path):
            continue
            
        results_df = pd.read_csv(result_path)
        races_dict = load_race_cards(race_card_path, motor_df)
        odds_dict = load_preview_odds(preview_odds_path)
        
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
            
            wave_height = 4  
            wind_speed = 2   
            condition_type = judge_race_condition(wave_height, wind_speed)
            
            scores = calculate_boat_scores(boats, condition_type, stadium_id)
            allocated_bets, race_type = generate_target_return_bets(scores, race_actual_odds)
            
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
    print(f" 🎯 2026年 8月度 月間一括バックテスト最終結果（全レース完全購入版）")
    print("="*50)
    for r_type, st in type_stats.items():
        t_roi = (st["pay"] / st["inv"] * 100) if st["inv"] > 0 else 0
        t_hit = (st["hits"] / st["count"] * 100) if st["count"] > 0 else 0
        print(f"■ 【{r_type}】 レース数: {st['count']} | 的中数: {st['hits']} ({t_hit:.1f}%) | 投資: {st['inv']:,}円 | 払戻: {st['pay']:,}円 | 回収率: {t_roi:.1f}%")
    print("-" * 50)
    print(f" 見送りレース数 : {skipped_races} レース")
    print(f" 購入レース数   : {total_races} レース")
    print(f" 的中レース数   : {hit_count} レース (的中率: {hit_rate:.2f}%)")
    print(f" 総投資額       : {total_investment:,.0f} 円")
    print(f" 総払戻金       : {total_payout:,.0f} 円")
    print(f" 収支           : {net_profit:+,.0f} 円")
    print(f" 総合回収率 (ROI) : {roi:.2f}%")
    print("="*50)

if __name__ == "__main__":
    run_monthly_backtest("2026-08-01", "2026-08-31")

