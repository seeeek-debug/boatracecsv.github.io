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

def load_original_exhibition(year, month, day_str):
    # 修正: 会場別ではなく日別のCSVファイル（例: 02.csv）を取得する
    path = f"data/previews/original_exhibition/{year}/{month}/{day_str}.csv"
    df = fetch_github_csv(path)
    return df

def evaluate_from_exhibition_times(motor_2ren, val1, val2, val3):
    deashi_score = 0
    if val2 > 0:
        if val2 <= 5.45: deashi_score += 2
        elif val2 <= 5.55: deashi_score += 1
    if val1 > 0:
        if val1 <= 36.8: deashi_score += 2
        elif val1 <= 37.3: deashi_score += 1

    if deashi_score >= 3: deashi_eval = "🔥S"
    elif deashi_score >= 2: deashi_eval = "⭐A+"
    elif deashi_score >= 1: deashi_eval = "✨A"
    else: deashi_eval = "⚖️B+"

    nobi_score = 0
    if val3 > 0:
        if val3 <= 6.65: nobi_score += 2
        elif val3 <= 6.78: nobi_score += 1

    if nobi_score >= 2: nobi_eval = "🔥S"
    elif nobi_score >= 1: nobi_eval = "⭐A+"
    else: nobi_eval = "✨A"

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
    else: overall_rank = "🔄B"

    if nobi_score > deashi_score: foot_type = "🚀伸び足特化型"
    elif deashi_score > nobi_score: foot_type = "🌀出足・回り足型"
    else: foot_type = "⚖️正統派バランス型"

    return overall_rank, foot_type, deashi_eval, nobi_eval

def generate_race_tactical_advice(racer_data_list, in_rate):
    if not racer_data_list or len(racer_data_list) < 6:
        return "【⚡展開混戦】データ不足のためフラットな評価"

    boat1 = racer_data_list[0]
    boat2 = racer_data_list[1]
    boat4 = racer_data_list[3]
    boat5 = racer_data_list[4]
    boat6 = racer_data_list[5]

    b1_win = boat1["win_rate"]
    b1_motor = boat1["motor_2ren"]

    strong_outs = [d for d in racer_data_list[1:] if d["win_rate"] >= 6.0 or d["motor_2ren"] >= 45.0]

    if b1_win <= 4.0 or b1_motor <= 30.0:
        target_boat = strong_outs[0]["boat_no"] if strong_outs else "2"
        return f"【⚠️ 1号艇ピンチ・波乱警戒】 1号艇の勝率・機力に不安あり。**{target_boat}号艇**の逆転・差し抜けに要警戒！"
    
    elif boat2["win_rate"] >= 5.8 and boat2["motor_2ren"] >= 38.0:
        return f"【🎯2号艇の差し鋭い】 2号艇({boat2['r_name']})の勝率・機力が高く、1号艇の懐を突く差し抜け・逆転展開に要注目！"

    elif boat4["win_rate"] >= 6.0 and boat4["motor_2ren"] >= 38.0:
        return f"【🚀4号艇のまくり一撃警戒】 4号艇({boat4['r_name']})の機力・実力が高く、カドからの自在なまくり・全速攻勢に注意！"

    elif (boat5["win_rate"] >= 5.5 or boat6["win_rate"] >= 5.5) and (boat5["motor_2ren"] >= 40.0 or boat6["motor_2ren"] >= 40.0):
        best_out = boat5 if (boat5["win_rate"] + boat5["motor_2ren"]*0.1) >= (boat6["win_rate"] + boat6["motor_2ren"]*0.1) else boat6
        return f"【🌐外枠(5・6号艇)の展開突き警戒】 アウト勢ながら実力・機力上位の**{best_out['boat_no']}号艇({best_out['r_name']})**が展開を突いて浮上するシーンに注意！"

    elif in_rate >= 55.0 and b1_win >= 5.5:
        return "【🛡️固め・イン鉄壁】 1号艇のイン逃げ信頼度高。相手探し（2・3号艇）が主軸。"
    
    else:
        return "【⚡差し・まくり交錯】 互角のメンバー構成。第1ターンマークの攻防に注目。"

def calculate_single_race_analysis(venue, venue_code, year, month, day_str, date_str, r):
    all_race_rates = load_race_course_win_rates()
    venue_rates_by_race = all_race_rates.get(venue, {})
    tendency = VENUE_TENDENCIES.get(venue, "標準水面")
    
    race_course_rate = venue_rates_by_race.get(r, {1: 50.0})
    in_rate = race_course_rate.get(1, 50.0)

    df_card, _ = load_race_card(venue, venue_code, year, month, day_str)
    df_exh = load_original_exhibition(year, month, day_str)

    summary_text = f"🏟️ **【{venue}場 R{r}】 AIレース分析 ({date_str})**\n"
    summary_text += f"📝 水面特性: *{tendency}* (1コース勝率: {in_rate:.1f}%)\n\n"
    
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

    # 日別展示データから「該当会場」と「該当レース」の行を絞り込む
    row_exh = None
    if df_exh is not None and not df_exh.empty:
        venue_col = next((c for c in df_exh.columns if "場" in c or "stadium" in c or "venue" in c), None)
        exh_r_col = next((c for c in df_exh.columns if "レース回" in c or "race" in c), None)
        
        if venue_col and exh_r_col:
            matched_exh = df_exh[
                df_exh[venue_col].astype(str).str.contains(venue, na=False) & 
                df_exh[exh_r_col].astype(str).str.contains(f"{r}R|{r}", na=False)
            ]
            if not matched_exh.empty:
                row_exh = matched_exh.iloc[0]

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

            try:
                avg_st = float(row_race.get(f"艇{b_no}_全国平均ST", 0.15))
            except:
                avg_st = 0.15

            val1, val2, val3 = 0.0, 0.0, 0.0
            if row_exh is not None:
                try:
                    val1 = float(row_exh.get(f"艇{b_no}_値1", 0.0))
                except:
                    pass
                try:
                    val2 = float(row_exh.get(f"艇{b_no}_値2", 0.0))
                except:
                    pass
                try:
                    val3 = float(row_exh.get(f"艇{b_no}_値3", 0.0))
                except:
                    pass

            overall_rank, foot_type, deashi_eval, nobi_eval = evaluate_from_exhibition_times(
                motor_2ren, val1, val2, val3
            )

            racer_structs.append({
                "boat_no": str(b_no),
                "r_name": r_name,
                "win_rate": win_rate,
                "motor_2ren": motor_2ren,
                "st": avg_st
            })

            eval_detail = (
                f"• **#{b_no} {r_name}** ({r_class}): 機力{overall_rank} ({foot_type})\n"
                f"   └ 勝率:{win_rate:.2f} | 出足:{deashi_eval} / 伸び:{nobi_eval}\n"
                f"   └ M#{motor_num} (2連:{motor_2ren:.1f}%) 展示[1周:{val1:.2f} / 回り足:{val2:.2f} / 直線:{val3:.2f}]"
            )
            racer_evals.append(eval_detail)

    tag = generate_race_tactical_advice(racer_structs, in_rate)
    summary_text += f"💡 **展開予想**: {tag}\n\n"
    
    if racer_evals:
        summary_text += "\n\n".join(racer_evals) + "\n"
        
    return summary_text

# --- 2段階セレクトメニューの定義 ---

class RaceSelect(discord.ui.Select):
    def __init__(self, venue):
        self.venue = venue
        options = [discord.SelectOption(label=f"第 {i} レース (R{i})", value=str(i), description=f"{venue}場 第{i}Rの分析を見る") for i in range(1, 13)]
        super().__init__(placeholder="🏁 分析するレースを選択してください...", min_values=1, max_values=1, options=options)

    async def callback(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)
        try:
            venue = self.venue
            venue_code = VENUE_MAPPING.get(venue, "01")
            r_num = int(self.values[0])
            
            target_date = datetime.now(JST)
            result_text = await asyncio.to_thread(
                calculate_single_race_analysis, venue, venue_code, 
                target_date.strftime("%Y"), target_date.strftime("%m"), 
                target_date.strftime("%d"), target_date.strftime("%Y-%m-%d"), r_num
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
        super().__init__(placeholder="🏟️ 会場を選択してください...", min_values=1, max_values=1, options=options)

    async def callback(self, interaction: discord.Interaction):
        venue = self.values[0]
        await interaction.response.send_message(
            content=f"🏟️ **【{venue}場】** が選択されました。\n続いて、分析したいレースを選んでください👇",
            view=RaceSelectView(venue),
            ephemeral=True
        )

class VenueSelectView(discord.ui.View):
    def __init__(self):
        super().__init__()
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
    if not daily_morning_report.is_running():
        daily_morning_report.start()

@bot.command(name="boat_report")
async def boat_report(ctx):
    await ctx.send("🏁 **【本日のAIレース分析】**\n下のメニューから会場を選んでください👇", view=VenueSelectView())

if __name__ == "__main__":
    keep_alive()
    token = os.environ.get("DISCORD_TOKEN")
    bot.run(token)

