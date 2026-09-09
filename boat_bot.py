import asyncio
from datetime import datetime, time, timezone, timedelta
import io
import os
import threading
from flask import Flask
import discord
from discord.ext import commands, tasks
import numpy as np
import pandas as pd
import requests

# --- Render用Webサーバー ---
app = Flask(__name__)

@app.route("/")
def home():
    return "I am alive!"

def run_web():
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, use_reloader=False)

def keep_alive():
    t = threading.Thread(target=run_web)
    t.daemon = True
    t.start()

# --- Discordボット設定 ---

GITHUB_RAW_BASE = "https://raw.githubusercontent.com/seeeek-debug/boatracecsv.github.io/main/"
NOTIFICATION_CHANNEL_ID = 1546042629253496925

JST = timezone(timedelta(hours=9))

VENUES = [
    "桐生", "戸田", "江戸川", "平和島", "多摩川", "浜名湖", 
    "蒲郡", "常滑", "津", "三国", "びわこ", "住之江", 
    "尼崎", "鳴門", "丸亀", "児島", "宮島", "徳山", 
    "下関", "若松", "芦屋", "福岡", "唐津", "大村"
]

VENUE_MAPPING = {
    "桐生": "01", "戸田": "02", "江戸川": "03", "平和島": "04", 
    "多摩川": "05", "浜名湖": "06", "蒲郡": "07", "常滑": "08", 
    "津": "09", "三国": "10", "びわこ": "11", "住之江": "12", 
    "尼崎": "13", "鳴門": "14", "丸亀": "15", "児島": "16", 
    "宮島": "17", "徳山": "18", "下関": "19", "若松": "20", 
    "芦屋": "21", "福岡": "22", "唐津": "23", "大村": "24"
}

VENUE_TENDENCIES = {
    "大村": "イン鉄板・静穏水面", "芦屋": "イン優勢・静穏", "徳山": "イン優勢・走りやすい",
    "桐生": "標高が高くモーターパワー重要", "戸田": "狭い水面・まくり差し多発", 
    "江戸川": "日本一の難水面・大荒れ警戒", "平和島": "まくり・差し交錯・イン苦戦",
    "多摩川": "静穏だが水面は軽め", "浜名湖": "広大な水面・スピード戦",
    "蒲郡": "ナイター・直線足重視", "常滑": "伊勢湾の風に注意", "津": "クセのない標準水面",
    "三国": "冬場は荒れやすい", "びわこ": "淡水特有のうねりと難解さ",
    "住之江": "ナイトプール・インと差しの攻防", "尼崎": "高速水面・イン信頼度高",
    "鳴門": "インが弱く激しい潮流", "丸亀": "ナイター・風の影響少",
    "児島": "潮の満ち引きで変化", "宮島": "瀬戸内の潮汐と難水面",
    "下関": "安定した水面コンディション", "若松": "海水面・うねり考慮",
    "福岡": "難水面・外マイ警戒", "唐津": "比較的フラットな水面"
}

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

CSV_CACHE = {}

def fetch_github_csv(file_path):
    if file_path in CSV_CACHE:
        return CSV_CACHE[file_path]
    url = f"{GITHUB_RAW_BASE}{file_path}"
    try:
        res = requests.get(url)
        if res.status_code == 200:
            df = pd.read_csv(io.StringIO(res.text))
            CSV_CACHE[file_path] = df
            return df
    except Exception as e:
        print(f"CSV Fetch Error ({file_path}): {e}")
    return None

def load_race_course_win_rates():
    df = fetch_github_csv("data/estimate/stadium/course_win_rate.csv")
    venue_race_rates = {}
    if df is not None and not df.empty:
        venue_col = "場コード" if "場コード" in df.columns else "stadium_code"
        race_col = "レース回" if "レース回" in df.columns else "race"
        inv_mapping = {int(v): k for k, v in VENUE_MAPPING.items()}
        for _, row in df.iterrows():
            try:
                v_code = int(row[venue_col])
                if v_code not in inv_mapping:
                    continue
                venue_name = inv_mapping[v_code]
                r_num = int(str(row[race_col]).replace("R", ""))
                course_rates = {}
                for c in range(1, 7):
                    col_name = f"{c}コース勝率"
                    if col_name in df.columns:
                        val = float(row[col_name])
                        course_rates[c] = val * 100 if val <= 1.0 else val
                    else:
                        course_rates[c] = 0.0
                if venue_name not in venue_race_rates:
                    venue_race_rates[venue_name] = {}
                venue_race_rates[venue_name][r_num] = course_rates
            except Exception:
                continue
    return venue_race_rates

def load_race_card(venue, venue_code, year, month, day_str):
    path = f"data/programs/race_cards/{year}/{month}/{day_str}.csv"
    df = fetch_github_csv(path)
    if df is not None and not df.empty:
        target_col = next((col for col in df.columns if any(k in col for k in ["場コード", "stadium_code", "場", "stadium", "venue"])), None)
        if target_col:
            df_filtered = df[
                df[target_col].astype(str).str.contains(venue_code, na=False) | 
                df[target_col].astype(str).str.contains(venue_code.lstrip('0'), na=False) |
                df[target_col].astype(str).str.contains(venue, na=False)
            ]
            if not df_filtered.empty:
                return df_filtered, path
        return df, path
    
    try:
        dt = datetime(int(year), int(month), int(day_str)) - timedelta(days=1)
        prev_path = f"data/programs/race_cards/{dt.strftime('%Y')}/{dt.strftime('%m')}/{dt.strftime('%d')}.csv"
        df_prev = fetch_github_csv(prev_path)
        if df_prev is not None and not df_prev.empty:
            target_col = next((col for col in df_prev.columns if any(k in col for k in ["場コード", "stadium_code", "場", "stadium", "venue"])), None)
            if target_col:
                df_filtered = df_prev[
                    df_prev[target_col].astype(str).str.contains(venue_code, na=False) | 
                    df_prev[target_col].astype(str).str.contains(venue_code.lstrip('0'), na=False) |
                    df_prev[target_col].astype(str).str.contains(venue, na=False)
                ]
                if not df_filtered.empty:
                    return df_filtered, prev_path
            return df_prev, prev_path
    except Exception:
        pass
    return None, path

def load_past_3months_original_exhibition(venue_code, year, month):
    dfs = []
    dt = datetime(int(year), int(month), 1)
    for _ in range(3):
        y = dt.strftime("%Y")
        m = dt.strftime("%m")
        path = f"data/previews/original_exhibition/{y}/{m}/{venue_code}.csv"
        df = fetch_github_csv(path)
        if df is not None and not df.empty:
            dfs.append(df)
        dt = (dt - timedelta(days=1)).replace(day=1)
        
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return None

def evaluate_relative_from_past_exhibition(boat_no, motor_2ren, win_rate, past_orig_df, r_num):
    if motor_2ren >= 48.0: motor_base = 10
    elif motor_2ren >= 43.0: motor_base = 9
    elif motor_2ren >= 39.0: motor_base = 8
    elif motor_2ren >= 35.0: motor_base = 7
    elif motor_2ren >= 32.0: motor_base = 6
    elif motor_2ren >= 29.0: motor_base = 5
    elif motor_2ren >= 26.0: motor_base = 4
    elif motor_2ren >= 23.0: motor_base = 3
    else: motor_base = 2

    if win_rate >= 7.8: win_base = 10
    elif win_rate >= 7.2: win_base = 9
    elif win_rate >= 6.8: win_base = 8
    elif win_rate >= 6.4: win_base = 7
    elif win_rate >= 6.0: win_base = 6
    elif win_rate >= 5.5: win_base = 5
    elif win_rate >= 5.0: win_base = 4
    else: win_base = 3

    orig_deashi_bonus = 0.0
    orig_nobi_bonus = 0.0

    if past_orig_df is not None and not past_orig_df.empty:
        try:
            race_col = next((c for c in past_orig_df.columns if "レース" in c or "race" in c.lower() or "R" in c), None)
            boat_col = next((c for c in past_orig_df.columns if "艇" in c or "boat" in c.lower() or "no" in c.lower()), None)
            
            if race_col and boat_col:
                matched_orig = past_orig_df[
                    (past_orig_df[race_col].astype(str).str.contains(str(r_num))) & 
                    (past_orig_df[boat_col].astype(str).str.contains(str(boat_no)))
                ]
                if not matched_orig.empty:
                    val_cols = [c for c in matched_orig.columns if any(k in c for k in ["タイム", "評価", "time", "val", "score"])]
                    if val_cols:
                        for col in val_cols:
                            numeric_vals = pd.to_numeric(matched_orig[col], errors='coerce').dropna()
                            if not numeric_vals.empty:
                                orig_deashi_bonus += 0.3
                                orig_nobi_bonus += 0.4
        except Exception as e:
            print(f"Original exhibition matching error: {e}")

    deashi_val = motor_base * 0.6 + win_base * 0.4 + orig_deashi_bonus
    if boat_no in [1, 2]:
        deashi_val += 0.6
    elif boat_no >= 5:
        deashi_val -= 0.4
    deashi_score = int(np.clip(round(deashi_val), 1, 10))

    nobi_val = motor_base * 0.5 + win_base * 0.3 + orig_nobi_bonus
    if boat_no in [4, 5, 6]:
        nobi_val += 1.0
    elif boat_no == 1:
        nobi_val -= 0.4
    nobi_score = int(np.clip(round(nobi_val), 1, 10))

    overall_base = (deashi_score * 0.55 + nobi_score * 0.45)
    overall_score = int(np.clip(round(overall_base), 1, 10))

    rank_stars = "★" * overall_score + "☆" * (10 - overall_score)
    overall_rank = f"【{overall_score} / 10】 {rank_stars}"

    deashi_eval = f"【{deashi_score}/10】"
    nobi_eval = f"【{nobi_score}/10】"

    if nobi_score > deashi_score: foot_type = "伸び足型"
    elif deashi_score > nobi_score: foot_type = "出足型"
    else: foot_type = "バランス型"

    return overall_rank, foot_type, deashi_eval, nobi_eval, overall_score, deashi_score, nobi_score

def calculate_course_probabilities(race_course_rate, racer_structs):
    """CSVに入っている全コース勝率データをベースに、選手勝率・機力で微調整して1着率を算出"""
    raw_scores = []
    for i, b in enumerate(racer_structs):
        b_no = i + 1  # 1〜6コース
        # CSVから取得した各コースの基本勝率（デフォルトは一律10%等）
        base_rate = race_course_rate.get(b_no, 10.0)
        
        # 選手の能力・機力指数（平均的な選手なら1.0前後）
        power_factor = (b["win_rate"] * 0.5 + b["overall_score"] * 0.5) / 6.0
        
        # CSVのコース勝率をベースに、選手の力で微調整
        score = base_rate * max(power_factor, 0.4)
        raw_scores.append(max(score, 0.1))

    # 合計を100%に正規化
    total_score = sum(raw_scores)
    win_probs = [s / total_score * 100 for s in raw_scores]

    # 2連率・3連率の算出（1着率をベースに競艇の統計的傾向に沿って上乗せ）
    double_probs = []
    triple_probs = []
    for i, p1 in enumerate(win_probs):
        b_no = i + 1
        if b_no == 1:
            d_rate = min(p1 + 35.0, 92.0)
            t_rate = min(d_rate + 20.0, 98.0)
        elif b_no == 2:
            d_rate = min(p1 + 25.0, 75.0)
            t_rate = min(d_rate + 22.0, 88.0)
        elif b_no == 3:
            d_rate = min(p1 + 18.0, 60.0)
            t_rate = min(d_rate + 20.0, 78.0)
        elif b_no == 4:
            d_rate = min(p1 + 12.0, 48.0)
            t_rate = min(d_rate + 18.0, 68.0)
        elif b_no == 5:
            d_rate = min(p1 + 8.0, 35.0)
            t_rate = min(d_rate + 15.0, 55.0)
        else:
            d_rate = min(p1 + 5.0, 25.0)
            t_rate = min(d_rate + 12.0, 42.0)
            
        double_probs.append(d_rate)
        triple_probs.append(t_rate)

    return win_probs, double_probs, triple_probs

def generate_race_tactical_advice(racer_data_list, in_rate, venue_name):
    if not racer_data_list or len(racer_data_list) < 6:
        return "【⚠️ 展開混戦】データ不足のためフラットな評価", "混戦", "特になし", "特になし"

    b1 = racer_data_list[0]
    b2 = racer_data_list[1]
    b3 = racer_data_list[2]
    b4 = racer_data_list[3]
    b5 = racer_data_list[4]
    b6 = racer_data_list[5]

    diff_2_1_deashi  = b2["deashi_score"] - b1["deashi_score"]
    diff_3_1_nobi    = b3["nobi_score"]   - b1["nobi_score"]

    is_strong_in_venue = venue_name in ["大村", "芦屋", "徳山"]

    next_kimarite_advice = "特になし"

    if b1["win_rate"] <= 4.2 or b1["overall_score"] <= 4 or diff_2_1_deashi >= 3:
        tag = f"【⚠️ 1号艇ピンチ】 1号艇足色劣勢（対2号艇出足差: {diff_2_1_deashi:+d}）。2号艇の差し・波乱警戒"
        recommended_kimarite = "差し / まくり"
        next_kimarite_advice = f"2号艇({b2['r_name']})の差し抜け (`2-1, 2-3`) または外マイ連動"
    elif diff_3_1_nobi >= 2 and b3["win_rate"] >= 6.0:
        tag = f"【⚡ 3号艇の自在攻め】 3号艇の伸び足が魅力（対1号艇伸び差: {diff_3_1_nobi:+d}）"
        recommended_kimarite = "まくり差し (1-3, 3-1)"
        next_kimarite_advice = f"3号艇のまくり差し追走 (`1-3-2`)、または3号艇頭の全流し"
    elif diff_2_1_deashi >= 2 and b2["win_rate"] >= 6.0:
        tag = f"【🎯 2号艇の差し鋭い】 2号艇の出足が光る（対1号艇出足差: {diff_2_1_deashi:+d}）"
        recommended_kimarite = "差し (2-1系)"
        next_kimarite_advice = f"2号艇が差した後の1号艇の残り(`2-1`)、3号艇の展開突き(`2-1-3` / `2-3-1`)"
    elif is_strong_in_venue and b1["win_rate"] >= 5.5 and diff_2_1_deashi < 2:
        tag = f"【🛡️ イン堅実】 {venue_name}の水面特性と1号艇の踏ん張り。1-2・1-3本線"
        recommended_kimarite = "逃げ (1-2, 1-3)"
        next_kimarite_advice = f"**【逃げ時の次位】** 2号艇の堅実な差し追走(`1-2`) または 3号艇の自在ハンドル(`1-3`)。ヒモは4号艇のマーク(`1-X-4`)"
    elif in_rate >= 52.0 and b1["win_rate"] >= 5.5 and diff_2_1_deashi <= 0:
        tag = f"【🛡️ イン鉄壁ムード】 1号艇の出足が優勢（出足差: {diff_2_1_deashi:+d}）"
        recommended_kimarite = "逃げ (1-2)"
        next_kimarite_advice = f"**【逃げ時の次位】** 2号艇の差し(`1-2`)が本線。スリット同体なら1-2-3、2号艇が遅れれば1-3-2"
    else:
        tag = f"【⚔️ 展開もつれ】 機力拮抗でヒモ荒れ注意"
        recommended_kimarite = "差し / 1-2-3"
        next_kimarite_advice = f"混戦模様のため、ボックスや流し（`1-23-234`）推奨"

    longshot_items = []
    for b in racer_data_list:
        b_no = int(b["boat_no"])
        if b_no >= 4 and (b["nobi_score"] >= 7 or b["overall_score"] >= 7):
            if b["nobi_score"] > b["deashi_score"]:
                longshot_items.append(f"**{b_no}号艇 ({b['r_name']})** のセンター・大外まくり (`{b_no}-全-全`)")
            else:
                longshot_items.append(f"**{b_no}号艇 ({b['r_name']})** の展開突き・差し抜け (`{b_no}着ケツづまり狙い`)")
                
    if b2["deashi_score"] >= 8 and b1["overall_score"] <= 6:
        longshot_items.append(f"**2号艇 ({b2['r_name']})** の鋭い差し抜け (`2-1, 2-3`)")
    if b3["nobi_score"] >= 8 and b1["overall_score"] <= 6:
        longshot_items.append(f"**3号艇 ({b3['r_name']})** のまくり強襲 (`3-1, 3-4`)")

    if longshot_items:
        longshot_advice = " / ".join(longshot_items[:2])
    else:
        longshot_advice = "目立った特大気配の穴党向け伏兵は不在。手堅い決着が本線"

    return tag, recommended_kimarite, next_kimarite_advice, longshot_advice

def calculate_single_race_analysis(venue, venue_code, year, month, day_str, date_str, r):
    all_race_rates = load_race_course_win_rates()
    venue_rates_by_race = all_race_rates.get(venue, {})
    tendency = VENUE_TENDENCIES.get(venue, "標準水面")
    
    race_course_rate = venue_rates_by_race.get(r, {1: 50.0, 2: 15.0, 3: 12.0, 4: 10.0, 5: 8.0, 6: 5.0})
    in_rate = race_course_rate.get(1, 50.0)

    df_card, _ = load_race_card(venue, venue_code, year, month, day_str)
    past_orig_df = load_past_3months_original_exhibition(venue_code, year, month)

    summary_text = f"🏟️ **【{venue}場 R{r}】 AIレース分析 ({date_str})**\n"
    summary_text += f"📝 水面特性: *{tendency}* (1コース勝率: {in_rate:.1f}%)\n"
    summary_text += f"📊 評価方式: モーター素性 ＋ 過去オリジナル展示平均値 ＋ **全コース勝率データ連動**\n"
    summary_text += "━━━━━━━━━━━━━━━━━━━━━━\n"
    
    if df_card is None or df_card.empty:
        return summary_text + f"⚠️ 出走表データが取得できませんでした。"

    col_r_num = "レース回" if "レース回" in df_card.columns else ("レース" if "レース" in df_card.columns else None)
    
    row_race = None
    if col_r_num:
        matched = df_card[df_card[col_r_num].astype(str).str.contains(f"{r}R|{r}")]
        if not matched.empty:
            row_race = matched.iloc[0]
    else:
        if len(df_card) >= r:
            row_race = df_card.iloc[r-1]

    racer_evals = []
    racer_structs = []
    
    if row_race is not None:
        for b_no in range(1, 7):
            r_name = str(row_race.get(f"艇{b_no}_選手名", f"{b_no}号艇"))
            r_class = str(row_race.get(f"艇{b_no}_級別", "B1"))
            
            try:
                win_rate = float(row_race.get(f"艇{b_no}_全国勝率", 0.0))
            except:
                win_rate = 0.0
                
            try:
                motor_num = int(row_race.get(f"艇{b_no}_モーター番号", 0))
            except:
                motor_num = 0

            motor_2ren = 0.0
            try:
                motor_2ren = float(row_race.get(f"艇{b_no}_モーター2連対率", 0.0))
            except:
                pass
            
            overall_rank, foot_type, deashi_eval, nobi_eval, overall_score, deashi_score, nobi_score = evaluate_relative_from_past_exhibition(
                b_no, motor_2ren, win_rate, past_orig_df, r
            )

            racer_structs.append({
                "boat_no": str(b_no),
                "r_name": r_name,
                "r_class": r_class,
                "win_rate": win_rate,
                "motor_2ren": motor_2ren,
                "overall_score": overall_score,
                "deashi_score": deashi_score,
                "nobi_score": nobi_score,
                "st": float(row_race.get(f"艇{b_no}_全国平均ST", 0.15))
            })

            eval_detail = (
                f"**#{b_no} {r_name}** ({r_class})  |  機力評価: {overall_rank}\n"
                f"└ 勝率: **{win_rate:.2f}**  /  出足: {deashi_eval}  /  伸び: {nobi_eval} ({foot_type})\n"
                f"└ モーター: M#{motor_num} (2連対率: **{motor_2ren:.1f}%**)"
            )
            racer_evals.append(eval_detail)

    tag, recommended_kimarite, next_kimarite_advice, longshot_advice = generate_race_tactical_advice(racer_structs, in_rate, venue)
    
    # CSVから取得した各コース勝率データを渡して確率を算出
    win_p, double_p, triple_p = calculate_course_probabilities(race_course_rate, racer_structs)

    summary_text += f"💡 **展開予想**: {tag}\n"
    summary_text += f"🎯 **推奨決まり手**: `{recommended_kimarite}`\n"
    summary_text += f"🔄 **逃げ・本線時の次位展開**: {next_kimarite_advice}\n"
    summary_text += f"💥 **穴狙い目**: {longshot_advice}\n"
    summary_text += "━━━━━━━━━━━━━━━━━━━━━━\n"
    
    summary_text += "📈 **【各コースの予想確率 (1着 / 2連 / 3連)】**\n"
    for i, b in enumerate(racer_structs):
        summary_text += f"• **{b['boat_no']}コース** ({b['r_name']}) ➔ 1着: **{win_p[i]:.1f}%** | 2連: **{double_p[i]:.1f}%** | 3連: **{triple_p[i]:.1f}%**\n"
    summary_text += "━━━━━━━━━━━━━━━━━━━━━━\n\n"

    if racer_evals:
        summary_text += "\n\n".join(racer_evals) + "\n"
        
    return summary_text

# --- 永続セレクトメニューの定義 ---

class RaceSelect(discord.ui.Select):
    def __init__(self, venue):
        self.venue = venue
        options = [
            discord.SelectOption(label="🌐 全レース一括表示 (1R〜12R)", value="all", description=f"{venue}場の全12レースの展開予想を一挙に見る")
        ]
        options.extend([
            discord.SelectOption(label=f"第 {i} レース (R{i})", value=str(i), description=f"{venue}場 第{i}Rの詳細分析を見る") 
            for i in range(1, 13)
        ])
        super().__init__(placeholder="🏁 分析するレースを選択してください...", min_values=1, max_values=1, options=options)

    async def callback(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)
        try:
            venue = self.venue
            venue_code = VENUE_MAPPING.get(venue, "01")
            val = self.values[0]
            
            target_date = datetime.now(JST)
            year = target_date.strftime("%Y")
            month = target_date.strftime("%m")
            day_str = target_date.strftime("%d")
            date_str = target_date.strftime("%Y-%m-%d")

            if val == "all":
                all_summaries = [f"🏟️ **【{venue}場】 全12レース展開予想一覧 ({date_str})**\n━━━━━━━━━━━━━━━━━━━━━━"]
                
                all_race_rates = load_race_course_win_rates()
                venue_rates_by_race = all_race_rates.get(venue, {})
                df_card, _ = load_race_card(venue, venue_code, year, month, day_str)
                past_orig_df = load_past_3months_original_exhibition(venue_code, year, month)

                for r in range(1, 13):
                    race_course_rate = venue_rates_by_race.get(r, {1: 50.0, 2: 15.0, 3: 12.0, 4: 10.0, 5: 8.0, 6: 5.0})
                    in_rate = race_course_rate.get(1, 50.0)

                    racer_structs = []
                    if df_card is not None and not df_card.empty:
                        col_r_num = "レース回" if "レース回" in df_card.columns else ("レース" if "レース" in df_card.columns else None)
                        row_race = None
                        if col_r_num:
                            matched = df_card[df_card[col_r_num].astype(str).str.contains(f"{r}R|{r}")]
                            if not matched.empty:
                                row_race = matched.iloc[0]
                        
                        else:
                            if len(df_card) >= r:
                                row_race = df_card.iloc[r-1]

                        if row_race is not None:
                            for b_no in range(1, 7):
                                r_name = str(row_race.get(f"艇{b_no}_選手名", f"{b_no}号艇"))
                                try:
                                    win_rate = float(row_race.get(f"艇{b_no}_全国勝率", 0.0))
                                except:
                                    win_rate = 0.0
                                try:
                                    motor_2ren = float(row_race.get(f"艇{b_no}_モーター2連対率", 0.0))
                                except:
                                    motor_2ren = 0.0
                                
                                _, _, _, _, overall_score, deashi_score, nobi_score = evaluate_relative_from_past_exhibition(
                                    b_no, motor_2ren, win_rate, past_orig_df, r
                                )
                                racer_structs.append({
                                    "boat_no": str(b_no),
                                    "r_name": r_name,
                                    "win_rate": win_rate,
                                    "overall_score": overall_score,
                                    "deashi_score": deashi_score,
                                    "nobi_score": nobi_score
                                })

                    tag, recommended_kimarite, next_kimarite_advice, longshot_advice = generate_race_tactical_advice(racer_structs, in_rate, venue)
                    all_summaries.append(f"**【第{r}R】** {tag}  |  推: `{recommended_kimarite}`\n└ 次位: {next_kimarite_advice}\n└ 💥穴: {longshot_advice}")

                result_text = "\n".join(all_summaries)
            else:
                r_num = int(val)
                result_text = await asyncio.to_thread(
                    calculate_single_race_analysis, venue, venue_code, 
                    year, month, day_str, date_str, r_num
                )

            if len(result_text) <= 2000:
                await interaction.followup.send(content=result_text, ephemeral=True)
            else:
                for i in range(0, len(result_text), 2000):
                    await interaction.followup.send(content=result_text[i:i+2000], ephemeral=True)
        except Exception as e:
            print(f"Error: {e}")
            await interaction.followup.send(content=f"⚠️ エラーが発生しました: {e}", ephemeral=True)

class RaceSelectView(discord.ui.View):
    def __init__(self, venue):
        super().__init__()
        self.add_item(RaceSelect(venue))

class VenueSelect(discord.ui.Select):
    def __init__(self):
        options = [discord.SelectOption(label=v, description=f"{v}場のレースを選択する") for v in VENUES]
        super().__init__(placeholder="🏟️ 会場を選択してください...", min_values=1, max_values=1, options=options, custom_id="persistent_venue_select")

    async def callback(self, interaction: discord.Interaction):
        venue = self.values[0]
        await interaction.response.send_message(
            content=f"🏟️ **【{venue}場】** が選択されました。続いて、分析したいレースまたは全レース一覧を選んでください👇",
            view=RaceSelectView(venue),
            ephemeral=True
        )

class VenueSelectView(discord.ui.View):
    def __init__(self):
        super().__init__(timeout=None)
        self.add_item(VenueSelect())

@tasks.loop(time=time(hour=8, minute=30, tzinfo=JST))
async def daily_morning_report():
    channel = bot.get_channel(NOTIFICATION_CHANNEL_ID)
    if channel is not None:
        await channel.send("🏁 **【本日のAIレース分析】**\n下のメニューから会場を選んでください👇", view=VenueSelectView())

@daily_morning_report.before_loop
async def before_daily_report():
    await bot.wait_until_ready()

@bot.event
async def on_ready():
    print(f"Logged in as {bot.user.name}!")
    bot.add_view(VenueSelectView())
    if not daily_morning_report.is_running():
        daily_morning_report.start()

@bot.command(name="setup")
async def setup_menu(ctx):
    await ctx.send("🏁 **【AIレース分析メニュー】**\n下のメニューからいつでも会場を選んでください👇", view=VenueSelectView())
    await ctx.message.delete()

if __name__ == "__main__":
    keep_alive()
    token = os.environ.get("DISCORD_TOKEN")
    bot.run(token)
