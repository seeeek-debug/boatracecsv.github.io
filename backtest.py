import os
import sys
import json
import urllib.request
import pandas as pd
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

STADIUM_ID_TO_NAME = {
    1: "桐生", 2: "戸田", 3: "江戸川", 4: "平和島", 5: "多摩川", 6: "浜名湖",
    7: "蒲郡", 8: "常滑", 9: "津", 10: "三国", 11: "びわこ", 12: "住之江",
    13: "尼崎", 14: "鳴門", 15: "丸亀", 16: "児島", 17: "宮島", 18: "徳山",
    19: "下関", 20: "若松", 21: "芦屋", 22: "福岡", 23: "唐津", 24: "大村"
}

PROVEN_STADIUM_IDS = [4, 9, 12, 13, 15]
INITIAL_CAPITAL = 100000

def load_motor_abilities(file_path="data/estimate/motor_ability_score_v4.csv"):
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    return pd.DataFrame()

def load_sui_preview_data(file_path):
    if not os.path.exists(file_path):
        return {}
    df = pd.read_csv(file_path)
    sui_dict = {}
    for idx, row in df.iterrows():
        race_code = str(row.get('レースコード', '')).strip()
        if not race_code:
            continue
        try:
            wind_speed = float(row.get('風速(m)', 0)) if pd.notna(row.get('風速(m)')) else 0.0
        except ValueError:
            wind_speed = 0.0
        try:
            wave_cm = float(row.get('波の高さ(cm)', 0)) if pd.notna(row.get('波の高さ(cm)')) else 0.0
        except ValueError:
            wave_cm = 0.0
        sui_dict[race_code] = {'wind_speed': wind_speed, 'wave_cm': wave_cm}
    return sui_dict

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
            
            local_win = float(row.get(f"{prefix}當地勝率", 5.0)) if pd.notna(row.get(f"{prefix}當地勝率")) else 5.0
            local_two = float(row.get(f"{prefix}當地2連対率", 30.0)) if pd.notna(row.get(f"{prefix}當地2連対率")) else 30.0
            motor_two = float(row.get(f"{prefix}モーター2連対率", 30.0)) if pd.notna(row.get(f"{prefix}モーター2連対率")) else 30.0
            
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
                'local_win': local_win,
                'local_two': local_two,
                'motor_two': motor_two,
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
            st = float(row.get(f"艇{b}_スタート展示", 0.15)) if pd.notna(row.get(f"艇{b}_スタート展示")) else 0.15
            boat_stt[b] = {'st': st}
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
            val = float(row.get(f"艇{b}_値1", 0)) if pd.notna(row.get(f"艇{b}_値1", 0)) else 0.0
            boat_orig[b] = {'val1': val}
        orig_dict[race_code] = boat_orig
    return orig_dict

def generate_target_return_bets_custom(boat_data_list, race_actual_odds, stt_info, orig_info, stadium_id):
    boat_scores = {}
    for data in boat_data_list:
        boat = data['boat_number']
        ex_st = stt_info.get(boat, {}).get('st', 0.15) if stt_info else 0.15
        st_score = max(0.25 - ex_st, 0) * 25 if ex_st > 0 else -10.0 
        
        orig_bonus = 0.0
        if orig_info and boat in orig_info:
            val = orig_info[boat]['val1']
            if 35.0 <= val <= 40.0:
                orig_bonus = (40.0 - val) * 0.4
        
        local_bonus = (data['local_win'] - 5.0) * 1.5 + (data['local_two'] - 30.0) * 0.03
        motor_stat_bonus = (data['motor_two'] - 30.0) * 0.04
        
        stat_score = data['class_bonus'] + st_score - data['f_penalty'] + orig_bonus + local_bonus
        motor_score = (data['motor_power'] * 2.0) + motor_stat_bonus
        
        raw_power = stat_score + motor_score
        boat_scores[boat] = max(raw_power, 0.1)

    if not race_actual_odds or not boat_scores:
        return None

    if stadium_id == 9:
        sorted_scores = sorted(boat_scores.values(), reverse=True)
        if len(sorted_scores) >= 2:
            score_diff = sorted_scores[0] - sorted_scores[1]
            if score_diff < 0.15:
                return None

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
                p3_conditional = base_probs[b3] / remaining_after_p2 if remaining_after_p2 > 0 else 0.25
                    
                combo = f"{b1}-{b2}-{b3}"
                combo_probs[combo] = p1 * p2_conditional * p3_conditional
                
    total_cp = sum(combo_probs.values())
    if total_cp > 0:
        combo_probs = {k: v / total_cp for k, v in combo_probs.items()}

    if stadium_id in [9, 13]:  
        min_odds, max_odds = 15.0, 40.0
        min_ev = 1.20
        max_bets = 3
        bet_amount = 200
        require_boat1_win = True
    else:  
        min_odds, max_odds = 12.0, 50.0
        min_ev = 1.25
        max_bets = 1
        bet_amount = 400
        require_boat1_win = False

    valid_bets = []
    for combo, combo_prob in combo_probs.items():
        if combo not in race_actual_odds:
            continue
        
        if require_boat1_win and not combo.startswith("1-"):
            continue

        actual_odds = race_actual_odds[combo]
        if not (min_odds <= actual_odds <= max_odds):
            continue
        if combo_prob < 0.01:
            continue
            
        expected_value = combo_prob * actual_odds
        if expected_value >= min_ev:
            valid_bets.append((combo, combo_prob, actual_odds, expected_value))
                
    if not valid_bets:
        return None
        
    valid_bets.sort(key=lambda x: x[3], reverse=True)
    selected = valid_bets[:max_bets]
    
    return [(item[0], bet_amount, item[2]) for item in selected] if selected else None

def send_discord_notification(message):
    webhook_url = os.environ.get("DISCORD_WEBHOOK_URL")
    if not webhook_url:
        print("Discord Webhook URLが設定されていません。")
        return
    
    payload = {"content": message}
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        webhook_url,
        data=data,
        headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"}
    )
    
    try:
        with urllib.request.urlopen(req) as response:
            if response.status == 204:
                print("Discordへの通知が完了しました。")
            else:
                print(f"Discord通知レスポンス: {response.status}")
    except Exception as e:
        print(f"Discord通知エラー: {e}")

def run_monthly_backtest(start_date="2026-07-01", end_date="2026-08-31"):
    motor_df = load_motor_abilities()
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    
    trade_history = []
    stadium_stats = {s_id: {"count": 0, "hits": 0, "inv": 0, "pay": 0} for s_id in PROVEN_STADIUM_IDS}
    
    print(f"=== 7・8月2ヶ月間テスト ＆ 資金推移・ドローダウン分析 ({start_date} 〜 {end_date}) ===")
    
    for single_date in dates:
        year = single_date.strftime("%Y")
        month = single_date.strftime("%m")
        day = single_date.strftime("%d")
        
        result_path = f"data/results/payouts/{year}/{month}/{day}.csv"
        race_card_path = f"data/programs/race_cards/{year}/{month}/{day}.csv"
        preview_odds_path = f"data/previews/od3/{year}/{month}/{day}.csv"
        stt_path = f"data/previews/stt/{year}/{month}/{day}.csv"
        orig_path = f"data/previews/original_exhibition/{year}/{month}/{day}.csv"
        sui_preview_path = f"data/previews/sui/{year}/{month}/{day}.csv"
        
        if not os.path.exists(result_path) or not os.path.exists(race_card_path):
            continue
            
        results_df = pd.read_csv(result_path)
        races_dict = load_race_cards(race_card_path, motor_df)
        odds_dict = load_preview_odds(preview_odds_path)
        stt_dict = load_stt_data(stt_path)
        orig_dict = load_original_exhibition_data(orig_path)
        sui_dict = load_sui_preview_data(sui_preview_path)
        
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
                if 'レース場' in str(col) or 'stadium' in str(col).lower() or '場コード' in str(col):
                    try:
                        stadium_id = int(row.get(col, 12))
                    except ValueError:
                        pass
                    break
            
            if stadium_id not in PROVEN_STADIUM_IDS:
                continue
            
            sui_info = sui_dict.get(race_code, {})
            wind_speed = sui_info.get('wind_speed', 0.0)
            wave_cm = sui_info.get('wave_cm', 0.0)
            max_wave = 10.0 if stadium_id == 4 else 3.0
            
            if wind_speed >= 5.0 or wave_cm >= max_wave:
                continue
            
            winning_combo = ""
            if '3連単_組番' in row and pd.notna(row['3連単_組番']):
                winning_combo = str(row['3連単_組番']).strip().replace('=', '-')
            
            payout_yen = float(row.get('3連単_払戻金', 0)) if pd.notna(row.get('3連単_払戻金', 0)) else 0.0
            
            if race_code not in races_dict:
                continue
                
            allocated_bets = generate_target_return_bets_custom(
                races_dict[race_code], odds_dict.get(race_code, {}), stt_dict.get(race_code, {}), orig_dict.get(race_code, {}), stadium_id
            )
            
            if allocated_bets is None:
                continue
            
            investment = sum(amount for combo, amount, odds in allocated_bets)
            stadium_stats[stadium_id]["count"] += 1
            stadium_stats[stadium_id]["inv"] += investment
            
            payout_total = 0.0
            hit_in_race = False
            for combo, amount, odds in allocated_bets:
                if combo == winning_combo:
                    hit_in_race = True
                    payout_added = (payout_yen / 100) * amount
                    payout_total += payout_added
                    stadium_stats[stadium_id]["pay"] += payout_added
            
            if hit_in_race:
                stadium_stats[stadium_id]["hits"] += 1

            profit = payout_total - investment
            trade_history.append({
                'datetime': single_date,
                'race_code': race_code,
                'investment': investment,
                'payout': payout_total,
                'profit': profit
            })

    trade_history.sort(key=lambda x: (x['datetime'], x['race_code']))
    
    current_capital = INITIAL_CAPITAL
    peak_capital = INITIAL_CAPITAL
    max_drawdown_amount = 0
    max_drawdown_rate = 0.0
    min_capital = INITIAL_CAPITAL
    max_capital = INITIAL_CAPITAL

    total_inv = 0
    total_pay = 0

    for t in trade_history:
        current_capital += t['profit']
        total_inv += t['investment']
        total_pay += t['payout']
        
        if current_capital > max_capital:
            max_capital = current_capital
        if current_capital < min_capital:
            min_capital = current_capital
            
        if current_capital > peak_capital:
            peak_capital = current_capital
        
        drawdown_amount = peak_capital - current_capital
        if drawdown_amount > max_drawdown_amount:
            max_drawdown_amount = drawdown_amount
            
        if peak_capital > 0:
            drawdown_rate = (drawdown_amount / peak_capital) * 100
            if drawdown_rate > max_drawdown_rate:
                max_drawdown_rate = drawdown_rate

    roi = (total_pay / total_inv * 100) if total_inv > 0 else 0
    net_profit = total_pay - total_inv

    # コンソール出力
    print("\n" + "="*50)
    print(f" 🎯 資金推移・リスク分析結果 ({start_date} 〜 {end_date})")
    print("="*50)
    print(f" 初期資金       : {INITIAL_CAPITAL:,} 円")
    print(f" 最終資金       : {current_capital:,.0f} 円")
    print(f" 総純利益       : {net_profit:+,.0f} 円")
    print(f" 総合回収率(ROI): {roi:.2f}%")
    print(f" 期間中最高資金 : {max_capital:,.0f} 円")
    print(f" 期間中最安資金 : {min_capital:,.0f} 円")
    print(f" 最大ドローダウン(金額)   : -{max_drawdown_amount:,.0f} 円")
    print(f" 最大ドローダウン(下落率) : -{max_drawdown_rate:.2f} %")
    print("="*50)

    # Discordへ通知するメッセージを作成して送信
    discord_message = (
        "**【ボートレース バックテスト結果通知】**\n"
        "```text\n"
        f" 対象期間       : {start_date} 〜 {end_date}\n"
        f" 初期資金       : {INITIAL_CAPITAL:,} 円\n"
        f" 最終資金       : {current_capital:,.0f} 円\n"
        f" 総純利益       : {net_profit:+,.0f} 円\n"
        f" 総合回収率(ROI): {roi:.2f} %\n"
        f" 最大ドローダウン : -{max_drawdown_rate:.2f} % (-{max_drawdown_amount:,.0f} 円)\n"
        "```"
    )
    send_discord_notification(discord_message)

if __name__ == "__main__":
    run_monthly_backtest("2026-07-01", "2026-08-31")

