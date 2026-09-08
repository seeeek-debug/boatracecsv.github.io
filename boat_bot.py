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
    "蒲郡", "常滑", "津", "三国", "琵琶湖", "住之江", 
    "尼崎", "児島", "丸亀", "徳山", "下関", "若松", 
    "芦屋", "福岡", "唐津", "大村"
]

VENUE_MAPPING = {
    "桐生": "01", "戸田": "02", "江戸川": "03", "平和島": "04", 
    "多摩川": "05", "浜名湖": "06", "蒲郡": "07", "常滑": "08", 
    "津": "09", "三国": "10", "琵琶湖": "11", "住之江": "12", 
    "尼崎": "13", "児島": "14", "丸亀": "15", "徳山": "16", 
    "下関": "17", "若松": "18", "芦屋": "19", "福岡": "20", 
    "唐津": "21", "大村": "22"
}

VENUE_TENDENCIES = {
    "大村": "イン鉄板・静穏水面", "芦屋": "イン優勢・静穏", "徳山": "イン優勢・走りやすい",
    "桐生": "標高が高くモーターパワー重要", "戸田": "狭い水面・まくり差し多発", 
    "江戸川": "日本一の難水面・大荒れ警戒", "平和島": "まくり・差し交錯・イン苦戦",
    "多摩川": "静穏だが水面は軽め", "浜名湖": "広大な水面・スピード戦",
    "蒲郡": "ナイター・直線足重視", "常滑": "伊勢湾の風に注意", "津": "クセのない標準水面",
    "三国": "冬場は荒れやすい", "琵琶湖": "淡水特有のうねりと難解さ",
    "住之江": "ナイトプール・インと差しの攻防", "尼崎": "高速水面・イン信頼度高",
    "児島": "潮の満ち引きで変化", "丸亀": "ナイター・風の影響少",
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

def load_tokuten_hayami(venue_code, year, month, day):
    path = f"data/previews/tokuten_hayami/{year}/{month}/{venue_code}.csv"
    df = fetch_github_csv(path)
    if df is not None and not df.empty:
        return df
    
    try:
        dt = datetime(int(year), int(month), int(day)) - timedelta(days=1)
        prev_path = f"data/previews/tokuten_hayami/{dt.strftime('%Y')}/{dt.strftime('%m')}/{venue_code}.csv"
        df_prev = fetch_github_csv(prev_path)
        if df_prev is not None and not df_prev.empty:
            return df_prev
    except Exception:
        pass
    return None

def load_original_exhibition_stats(venue_code, year, month):
    path = f"data/previews/original_exhibition/{year}/{month}/{venue_code}.csv"
    df = fetch_github_csv(path)
    
    if df is None or df.empty:
        path = f"data/programs/motor_stats/{year}/{month}/{venue_code}.csv"
        df = fetch_github_csv(path)
        
    exhibition_dict = {}
    if df is not None and not df.empty:
        m_col = next((c for c in ["モーター番号", "モーター", "motor"] if c in df.columns), None)
        rate_col = next((c for c in ["2連対率", "2連率"] if c in df.columns), None)
        win_col = "勝率" if "勝率" in df.columns else None
        
        c_wari = next((c for c in ["回り足タイム", "まわり足タイム", "回り足", "まわり足"] if c in df.columns), None)
        c_choku = next((c for c in ["直線タイム", "直線"] if c in df.columns), None)
        c_tenji = next((c for c in ["展示タイム", "展示"] if c in df.columns), None)
        c_1shu = next((c for c in ["1周タイム", "一周タイム", "1周", "一周"] if c in df.columns), None)

        if m_col:
            for _, row in df.iterrows():
                try:
                    m_num = int(row[m_col])
                    rate = float(row[rate_col]) if rate_col and pd.notna(row[rate_col]) else 0.0
                    w_rate = float(row[win_col]) if win_col and pd.notna(row[win_col]) else 0.0
                    
                    avg_wari = float(row[c_wari]) if c_wari and pd.notna(row[c_wari]) else 0.0
                    avg_choku = float(row[c_choku]) if c_choku and pd.notna(row[c_choku]) else 0.0
                    avg_tenji = float(row[c_tenji]) if c_tenji and pd.notna(row[c_tenji]) else 6.75
                    avg_1shu = float(row[c_1shu]) if c_1shu and pd.notna(row[c_1shu]) else 37.0
                    
                    exhibition_dict[m_num] = {
                        "2ren": rate,
                        "win": w_rate,
                        "avg_wari": avg_wari,
                        "avg_choku": avg_choku,
                        "avg_tenji": avg_tenji,
                        "avg_1shu": avg_1shu
                    }
                except:
                    continue
    return exhibition_dict

def get_shobugake_condition(info):
    """ 得点早見データの各着時得点率から、何着条件かを判定する """
    try:
        junni = int(info.get("junni", 99))
    except:
        junni = 99
        
    if junni <= 12:
        return "✨安全圏（優出・勝負余裕）"
        
    try:
        t1 = float(info.get("t1", 0) or 0)
        t2 = float(info.get("t2", 0) or 0)
        t3 = float(info.get("t3", 0) or 0)
        t4 = float(info.get("t4", 0) or 0)
        
        # 一般的な準優ボーダー（6.00）を基準に判定
        if t4 >= 6.0:
            return "🎯4着条件（比較的クリア容易）"
        elif t3 >= 6.0:
            return "🔥3着条件（勝負駆け・要着順）"
        elif t2 >= 6.0:
            return "⚡2着条件（勝負駆け・上位必須）"
        elif t1 >= 6.0:
            return "⚠️1着勝負（絶体絶命の勝負駆け）"
        else:
            return "⚠️厳しい条件（他力・完走目標）"
    except:
        return "🔥勝負駆け"

def evaluate_from_exhibition_times(motor_2ren, avg_wari, avg_choku, avg_tenji, avg_1shu):
    deashi_score = 0
    if avg_wari > 0:
        if avg_wari <= 1.48: deashi_score += 2
        elif avg_wari <= 1.52: deashi_score += 1
    if avg_1shu > 0:
        if avg_1shu <= 36.5: deashi_score += 2
        elif avg_1shu <= 37.0: deashi_score += 1

    if deashi_score >= 3: deashi_eval = "🔥S"
    elif deashi_score >= 2: deashi_eval = "⭐A+"
    elif deashi_score >= 1: deashi_eval = "✨A"
    else: deashi_eval = "⚖️B+"

    nobi_score = 0
    if avg_choku > 0:
        if avg_choku <= 6.18: nobi_score += 2
        elif avg_choku <= 6.23: nobi_score += 1
    if avg_tenji > 0:
        if avg_tenji <= 6.68: nobi_score += 2
        elif avg_tenji <= 6.72: nobi_score += 1

    if nobi_score >= 3: nobi_eval = "🔥S"
    elif nobi_score >= 2: nobi_eval = "⭐A+"
    elif nobi_score >= 1: nobi_eval = "✨A"
    else: nobi_eval = "⚖️B+"

    time_score = deashi_score + nobi_score
    overall_score = 0
    if motor_2ren >= 45.0: overall_score += 3
    elif motor_2ren >= 38.0: overall_score += 2
    elif motor_2ren >= 32.0: overall_score += 1

    overall_score += time_score

    if overall_score >= 6: overall_rank = "🔥S"
    elif overall_score >= 4: overall_rank = "⭐A+"
    elif overall_score >= 3: overall_rank = "✨A"
    elif overall_score >= 2: overall_rank = "⚖️B+"
    elif overall_score >= 1: overall_rank = "🔄B"
    else: overall_rank = "⚠️C"

    if nobi_score > deashi_score: foot_type = "🚀伸び足特化型"
    elif deashi_score > nobi_score: foot_type = "🌀出足・回り足型"
    else: foot_type = "⚖️正統派バランス型"

    return overall_rank, foot_type, deashi_eval, nobi_eval

def generate_race_tactical_advice(racer_data_list, in_rate):
    if not racer_data_list or len(racer_data_list) < 6:
        return "【⚡展開混戦】データ不足のためフラットな評価"

    boat1 = racer_data_list[0]
    b1_win = boat1["win_rate"]
    b1_motor = boat1["motor_2ren"]
    b1_shobu = boat1.get("shobu", "")

    strong_outs = [d for d in racer_data_list[1:] if d["win_rate"] >= 6.0 or d["motor_2ren"] >= 45.0]
    shobugake_boats = [d for d in racer_data_list if "勝負駆け" in d.get("shobu", "") or "1着勝負" in d.get("shobu", "")]

    if "1着勝負" in b1_shobu or "2着条件" in b1_shobu:
        return f"🔥【1号艇が勝負駆け・気迫の逃げ】 1号艇が崖っぷちの勝負駆け条件。是が非でもスタートを決めて逃げ切る構えに注目！"

    if shobugake_boats:
        sb_str = "・".join([f"{d['boat_no']}号艇({d['r_name']})" for d in shobugake_boats])
        return f"🔥【勝負駆け参戦レース】 {sb_str}が勝負駆け条件を抱えており、着順アップを狙う強気の攻め・思い切ったターンに警戒！"

    if b1_win <= 4.0 or b1_motor <= 30.0:
        target_boat = strong_outs[0]["boat_no"] if strong_outs else "2"
        return f"【⚠️ 1号艇ピンチ・波乱警戒】 1号艇の勝率・機力に不安あり。**{target_boat}号艇**の逆転・差し抜けに要警戒！"
    
    elif racer_data_list[2]["win_rate"] >= 6.5 and racer_data_list[2]["st"] <= 0.14:
        return f"【🌀3号艇のセンター攻め警戒】 3号艇の勝率が高く、全速まくり・まくり差し炸裂の展開に注意！"
    
    elif racer_data_list[3]["win_rate"] >= 6.5:
        return f"【🌀4号艇のカド自在戦警戒】 4号艇の実力が高く、カドからのダッシュ攻勢に要注目。"
    
    elif in_rate >= 55.0 and b1_win >= 5.5:
        return f"【🛡️固め・イン鉄壁】 1号艇のイン逃げ信頼度高。相手探し（2・3号艇）が主軸。"
    
    else:
        return f"【⚡差し・まくり交錯】 互角のメンバー構成。第1ターンマークの攻防に注目。"

def heavy_calculation(venue, venue_code, year, month, day, date_str):
    all_race_rates = load_race_course_win_rates()
    venue_rates_by_race = all_race_rates.get(venue, {})
    
    tendency = VENUE_TENDENCIES.get(venue, "標準水面")
    
    race_card_path = f"data/programs/race_cards/{year}/{month}/{venue_code}.csv"
    df_card = fetch_github_csv(race_card_path)
    exhibition_stats = load_original_exhibition_stats(venue_code, year, month)
    df_tokuten = load_tokuten_hayami(venue_code, year, month, day)
    
    tokuten_dict = {}
    if df_tokuten is not None and not df_tokuten.empty:
        for _, row in df_tokuten.iterrows():
            for i in range(1, 7):
                for prefix in [f"組{i}_", f"艇{i}_"]:
                    name_col = f"{prefix}選手名"
                    if name_col in df_tokuten.columns:
                        name = str(row.get(name_col, ""))
                        if name and name != "nan":
                            tokuten_dict[name] = {
                                "tokuten_ritsu": row.get(f"{prefix}得点率", 0.0),
                                "junni": row.get(f"{prefix}順位", "-"),
                                "border": row.get(f"{prefix}ボーダー状態", ""),
                                "t1": row.get(f"{prefix}1着時得点率", 0),
                                "t2": row.get(f"{prefix}2着時得点率", 0),
                                "t3": row.get(f"{prefix}3着時得点率", 0),
                                "t4": row.get(f"{prefix}4着時得点率", 0),
                            }

    summary_text = f"🏟️ **【{venue}場】 勝負駆け条件・展示評価 AI分析 ({date_str})**\n"
    summary_text += f"📝 水面特性: *{tendency}* | 📊 得点早見: *{'連携完了 ✅' if tokuten_dict else 'データなし ℹ️'}*\n\n"
    
    if df_card is None or df_card.empty:
        return summary_text + f"⚠️ 指定日の出走表データ（{race_card_path}）が取得できませんでした。"

    summary_text += "📋 **【レース別展開予測 ＆ 勝負駆け・機力詳細】**\n"
    
    col_r_num = "レース回" if "レース回" in df_card.columns else ("レース" if "レース" in df_card.columns else None)
    
    for r in range(1, 13):
        race_course_rate = venue_rates_by_race.get(r, {1: 50.0})
        in_rate = race_course_rate.get(1, 50.0)

        racer_evals = []
        racer_structs = []
        
        row_race = None
        if col_r_num:
            target_r_str = f"{r}R"
            matched = df_card[df_card[col_r_num].astype(str).str.contains(target_r_str)]
            if not matched.empty:
                row_race = matched.iloc[0]
        else:
            if len(df_card) >= r:
                row_race = df_card.iloc[r-1]

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

                avg_wari, avg_choku, avg_tenji, avg_1shu = 0.0, 0.0, 6.75, 37.0
                if motor_num in exhibition_stats:
                    if motor_2ren == 0.0:
                        motor_2ren = exhibition_stats[motor_num]["2ren"]
                    avg_wari = exhibition_stats[motor_num]["avg_wari"]
                    avg_choku = exhibition_stats[motor_num]["avg_choku"]
                    avg_tenji = exhibition_stats[motor_num]["avg_tenji"]
                    avg_1shu = exhibition_stats[motor_num]["avg_1shu"]

                try:
                    avg_st = float(row_race.get(f"艇{b_no}_全国平均ST", 0.15))
                except:
                    avg_st = 0.15

                overall_rank, foot_type, deashi_eval, nobi_eval = evaluate_from_exhibition_times(
                    motor_2ren, avg_wari, avg_choku, avg_tenji, avg_1shu
                )

                tokuten_info = tokuten_dict.get(r_name, {})
                shobu_cond = get_shobugake_condition(tokuten_info)
                t_ritsu = tokuten_info.get("tokuten_ritsu", None)
                t_junni = tokuten_info.get("junni", None)
                
                tokuten_str = ""
                if t_ritsu is not None and pd.notna(t_ritsu):
                    tokuten_str = f" [得点率:{float(t_ritsu):.2f}/順位:{t_junni}位]"

                racer_structs.append({
                    "boat_no": str(b_no),
                    "r_name": r_name,
                    "win_rate": win_rate,
                    "motor_2ren": motor_2ren,
                    "st": avg_st,
                    "shobu": shobu_cond
                })

                eval_detail = (
                    f"#{b_no} {r_name} ({r_class}): 機力{overall_rank}({foot_type}) | **{shobu_cond}**\n"
                    f"   └ └ 勝率:{win_rate:.2f}{tokuten_str} | 出足:{deashi_eval} / 伸び:{nobi_eval}\n"
                    f"   └ └ M#{motor_num}(2連:{motor_2ren:.1f}%) 展示平均[回り:{avg_wari:.2f}/直線:{avg_choku:.2f}/展示:{avg_tenji:.2f}/1周:{avg_1shu:.2f}]"
                )
                racer_evals.append(f"• {eval_detail}")

        tag = generate_race_tactical_advice(racer_structs, in_rate)
            
        if racer_evals:
            evals_str = "\n   ".join(racer_evals)
            summary_text += f"・ **R{r:2d}** (1ｺｰｽ勝率:{in_rate:.1f}%) ➔ {tag}\n   {evals_str}\n\n"
        else:
            summary_text += f"・ **R{r:2d}** (1ｺｰｽ勝率:{in_rate:.1f}%) ➔ {tag}\n\n"
            
    return summary_text

class VenueSelect(discord.ui.Select):
    def __init__(self):
        options = [discord.SelectOption(label=v, description=f"{v}場の勝負駆け・機力評価をAI分析") for v in VENUES]
        super().__init__(placeholder="🏟️ 詳細を確認したい会場を選択してください...", min_values=1, max_values=1, options=options)

    async def callback(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)
        
        venue = self.values[0]
        venue_code = VENUE_MAPPING.get(venue, "01")
        
        target_date = datetime.now(JST)
        year = target_date.strftime("%Y")
        month = target_date.strftime("%m")
        day = target_date.strftime("%d")
        date_str = target_date.strftime("%Y-%m-%d")
        
        summary_text = await asyncio.to_thread(
            heavy_calculation, venue, venue_code, year, month, day, date_str
        )

        await interaction.followup.send(content=summary_text, ephemeral=True)

class VenueSelectView(discord.ui.View):
    def __init__(self):
        super().__init__()
        self.add_item(VenueSelect())

@tasks.loop(time=time(hour=8, minute=30, tzinfo=JST))
async def daily_morning_report():
    channel = bot.get_channel(NOTIFICATION_CHANNEL_ID)
    if channel is not None:
        header = "🏁 **【毎朝の自動AIスクリーニング速報（勝負駆け条件完全対応版）】** 🏁\n下のメニューから気になる会場を選んで詳細をチェックしてな👇"
        await channel.send(header, view=VenueSelectView())

@daily_morning_report.before_loop
async def before_daily_report():
    await bot.wait_until_ready()

@bot.event
async def on_ready():
    print(f"Logged in as {bot.user.name}! Morning report active.")
    if not daily_morning_report.is_running():
        daily_morning_report.start()

@bot.command(name="boat_report")
async def boat_report(ctx):
    header = "🏁 **【全場AIスクリーニング速報（勝負駆け条件完全対応版）】** 🏁\n下のメニューから会場を選んで詳細をチェック👇"
    await ctx.send(header, view=VenueSelectView())

if __name__ == "__main__":
    keep_alive()
    token = os.environ.get("DISCORD_TOKEN")
    bot.run(token)

