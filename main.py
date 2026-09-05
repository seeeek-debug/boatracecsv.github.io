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
        # 出走表側のレースコードを特定（列名が異なる場合も考慮）
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
            
            class_bonus = {"A1": 1.2, "A2": 0.8, "B1": 0.4, "B2": 0.0}.get(class_type, 0.0)
            f_penalty = f_count * 0.5  
            
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

def judge_race_condition(wave_height, wind_speed):
    if wave_height <= 5 and wind_speed <= 3:
        return "solid"
    elif wave_height <= 10 and wind_speed <= 5:
        return "medium"
    else:
        return "rough"

def calculate_boat_scores(boat_data_list, condition_type):
    scores = {}
    if condition_type == "solid":
        w_frame, w_stat, w_motor = 2.0, 1.3, 1.1
    elif condition_type == "medium":
        w_frame, w_stat, w_motor = 1.3, 1.2, 1.4
    else:
        w_frame, w_stat, w_motor = 0.5, 1.0, 2.2

    for data in boat_data_list:
        boat = data['boat_number']
        frame_bias = (7 - boat) if condition_type != "rough" else (boat if boat >= 4 else 3)
        st_score = max(0.25 - data['avg_st'], 0) * 10
        stat_score = data['class_bonus'] + st_score - data['f_penalty']
        motor_score = data['motor_power'] * w_motor
        
        total_score = (frame_bias * w_frame) + (stat_score * w_stat) + motor_score
        scores[boat] = max(total_score, 0.1)
        
    return scores

def generate_sanrentan_bets(scores, condition_type):
    total_score = sum(scores.values())
    probs = {boat: score / total_score for boat, score in scores.items()}
    
    bets = []
    for b1 in range(1, 7):
        for b2 in range(1, 7):
            if b2 == b1: continue
            for b3 in range(1, 7):
                if b3 == b1 or b3 == b2: continue
                p1 = probs[b1]
                p2 = probs[b2] / (1 - p1) if (1 - p1) > 0 else 0
                p3 = probs[b3] / (1 - p1 - probs[b2]) if (1 - p1 - probs[b2]) > 0 else 0
                combo_prob = p1 * p2 * p3
                bets.append((f"{b1}-{b2}-{b3}", combo_prob))
                
    bets.sort(key=lambda x: x[1], reverse=True)
    
    if condition_type == "solid":
        return bets[:5]
    elif condition_type == "medium":
        return bets[:7]
    else:
        return [bet for bet in bets if not bet[0].startswith("1")][:6]

def run_backtest(target_date_str="20260901"):
    year = target_date_str[:4]
    month = target_date_str[4:6]
    day = target_date_str[6:]
    
    result_path = f"data/results/payouts/{year}/{month}/{day}.csv"
    race_card_path = f"data/programs/race_cards/{year}/{month}/{day}.csv"
    
    if not os.path.exists(result_path):
        print(f"結果データが見つかりません: {result_path}")
        return
    if not os.path.exists(race_card_path):
        print(f"出走表データが見つかりません: {race_card_path}")
        return

    results_df = pd.read_csv(result_path)
    motor_df = load_motor_abilities()
    races_dict = load_race_cards(race_card_path, motor_df)
    
    total_investment = 0
    total_payout = 0
    hit_count = 0
    total_races = 0
    
    print(f"=== 実データバックテスト実行中 ({target_date_str}) ===")
    
    for idx, row in results_df.iterrows():
        race_code = str(row.get('レースコード', ''))
        winning_combo = str(row.get('3連単_着順', ''))
        payout_yen = float(row.get('3連単_払戻金', 0))
        
        # レースコードが一致する出走データを取得
        if race_code not in races_dict:
            continue
            
        boats = races_dict[race_code]
        
        wave_height = 4  
        wind_speed = 2   
        condition_type = judge_race_condition(wave_height, wind_speed)
        
        scores = calculate_boat_scores(boats, condition_type)
        recommended_bets = generate_sanrentan_bets(scores, condition_type)
        
        investment = len(recommended_bets) * 100
        total_investment += investment
        total_races += 1
        
        hit = False
        for combo, prob in recommended_bets:
            if combo == winning_combo:
                hit = True
                total_payout += (payout_yen / 100) * 100
                break
        
        if hit:
            hit_count += 1
            print(f"[{race_code}] 【的中】 正解: {winning_combo} | 払戻: {payout_yen}円")
        else:
            print(f"[{race_code}] 【不的中】 正解: {winning_combo}")

    roi = (total_payout / total_investment * 100) if total_investment > 0 else 0
    hit_rate = (hit_count / total_races * 100) if total_races > 0 else 0
    net_profit = total_payout - total_investment
    
    print("\n" + "="*30)
    print(f" 🎯 バックテスト最終結果")
    print("="*30)
    print(f" 検証レース数 : {total_races} レース")
    print(f" 的中レース数 : {hit_count} レース (的中率: {hit_rate:.2f}%)")
    print(f" 総投資額     : {total_investment:,.0f} 円")
    print(f" 総払戻金     : {total_payout:,.0f} 円")
    print(f" 収支         : {net_profit:+,.0f} 円")
    print(f" 回収率 (ROI) : {roi:.2f}%")
    print("="*30)

if __name__ == "__main__":
    run_backtest("20260901")

