from datetime import datetime, time, timezone, timedelta
import io
import os
import discord
from discord.ext import commands, tasks
import numpy as np
import pandas as pd
import requests

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

def fetch_github_csv(file_path):
    url = f"{GITHUB_RAW_BASE}{file_path}"
    try:
        res = requests.get(url)
        if res.status_code == 200:
            return pd.read_csv(io.StringIO(res.text))
    except Exception as e:
        print(f"CSV Fetch Error ({file_path}): {e}")
    return None

def load_all_course_win_rates():
    df = fetch_github_csv("data/estimate/stadium/course_win_rate.csv")
    venue_course_rates = {}
    
    if df is not None and not df.empty:
        course_col = "コース" if "コース" in df.columns else "course"
        venue_col = "場コード" if "場コード" in df.columns else "stadium_code"
        win_col = "1着率" if "1着率" in df.columns else ("勝率" if "勝率" in df.columns else "win_rate")
        
        if course_col in df.columns and venue_col in df.columns and win_col in df.columns:
            inv_mapping = {int(v): k for k, v in VENUE_MAPPING.items()}
            for venue_code_val, group in df.groupby(venue_col):
                try:
                    code_int = int(venue_code_val)
                    if code_int in inv_mapping:
                        venue_name = inv_mapping[code_int]
                        course_rates = {}
                        for _, row in group.iterrows():
                            c = int(row[course_col])
                            r_val = float(row[win_col])
                            course_rates[c] = r_val * 100 if r_val <= 1.0 else r_val
                        venue_course_rates[venue_name] = course_rates
                except Exception:
                    continue
    return venue_course_rates

def analyze_vulnerability_trends(year, month, day):
    payout_path = f"data/results/payouts/{year}/{month}/{day}.csv"
    df_payout = fetch_github_csv(payout_path)
    
    if df_payout is None or df_payout.empty:
        return "⚠️ 本日の払戻・結果データがまだ公開されていません。"
    
    defeat_counts = {}
    col_name = "2連単_組番"
    
    if col_name in df_payout.columns:
        for val in df_payout[col_name].dropna():
            parts = str(val).replace("=", "-").split("-")
            if len(parts) >= 2:
                winner = parts[0]
                loser_side = parts[1]
                
                if winner != "1":
                    key = f"1コース敗北（{winner}コース被弾）"
                    defeat_counts[key] = defeat_counts.get(key, 0) + 1
                else:
                    key = f"1コース勝利時、2着粘り（{loser_side}号艇）"
                    defeat_counts[key] = defeat_counts.get(key, 0) + 1

    if not defeat_counts:
        return "・有効な組番データから敗因傾向を抽出できませんでした。"
        
    sorted_trends = sorted(defeat_counts.items(), key=lambda x: x[1], reverse=True)
    report_lines = []
    for trend, count in sorted_trends[:3]:
        report_lines.append(f"・ {trend}: **{count}件**")
        
    return "\n".join(report_lines)

def get_short_rank(score):
    if score >= 5.5:
        return "🔥S"
    elif score >= 4.5:
        return "⭐A+"
    elif score >= 3.5:
        return "✨A"
    elif score >= 2.5:
        return "⚖️B+"
    elif score >= 1.5:
        return "🔄B"
    else:
        return "⚠️C"

def calculate_historical_motor_score(current_year, current_month, current_day, venue_code, target_motor_no, days_back=60):
    current_date = datetime(int(current_year), int(current_month), int(current_day), tzinfo=JST)
    past_scores = []
    
    for i in range(1, days_back + 1, 2):
        past_date = current_date - timedelta(days=i)
        y = past_date.strftime("%Y")
        m = past_date.strftime("%m")
        d_str = past_date.strftime("%d")
        
        path = f"data/programs/race_cards/{y}/{m}/{d_str}.csv"
        df_past = fetch_github_csv(path)
        
        if df_past is not None and not df_past.empty:
            col_venue = "レース場コード" if "レース場コード" in df_past.columns else "場コード"
            if col_venue in df_past.columns:
                matched_venue = df_past[df_past[col_venue].astype(str).str.zfill(2) == str(venue_code)]
                for _, row in matched_venue.iterrows():
                    for b in range(1, 7):
                        m_col = f"艇{b}_モーター番号"
                        if m_col in row and str(row[m_col]) == str(target_motor_no):
                            q_col = f"艇{b}_モーター2連対率"
                            if q_col in row:
                                try:
                                    r_val = float(row[q_col])
                                    past_scores.append(r_val)
                                except:
                                    pass

    if not past_scores:
        return 3.0

    avg_metric = np.mean(past_scores)
    
    if avg_metric >= 45.0:
        return 6.0
    elif avg_metric >= 40.0:
        return 5.0
    elif avg_metric >= 35.0:
        return 4.0
    elif avg_metric >= 30.0:
        return 3.0
    elif avg_metric >= 25.0:
        return 2.0
    else:
        return 1.0

def generate_race_tactical_advice(racer_data_list, in_rate):
    if not racer_data_list or len(racer_data_list) < 6:
        return "【⚡展開混戦】データ不足のためフラットな評価"

    boat1 = racer_data_list[0]
    b1_class = boat1["class"]
    b1_st = boat1["st"]
    b1_f = boat1["f_count"]
    b1_motor = boat1["score"]

    boat3 = racer_data_list[2]
    boat4 = racer_data_list[3]

    strong_outs = []
    for d in racer_data_list[1:]:
        if "A1" in d["class"] or d["score"] >= 5.0 or (d["st"] <= 0.13 and d["st"] > 0):
            strong_outs.append(d)

    if b1_f > 0 or b1_motor <= 2.0 or b1_st >= 0.17:
        target_boat = strong_outs[0]["boat_no"] if strong_outs else "2"
        return f"【⚠️ 1号艇ピンチ・波乱警戒】 1号艇の不安あり。**{target_boat}号艇**の差し・まくり抜けに要警戒！"
    
    elif "A" in boat3["class"] and boat3["st"] <= 0.14 and boat3["score"] >= 4.0:
        return f"【🌀3号艇のまくり差し警戒】 3号艇({boat3['class']})の鋭い全速まくり差しが炸裂する展開に注意！"
    
    elif "A" in boat4["class"] and boat4["st"] <= 0.14 and boat4["score"] >= 4.0:
        return f"【🌀4号艇のまくり差し・カド攻め警戒】 4号艇({boat4['class']})のカドからの自在戦（まくり差し）に要注目。"
    
    elif in_rate >= 58.0 and "A" in b1_class and b1_motor >= 4.0:
        return f"【🛡️固め・イン鉄壁】 1号艇({b1_class})の逃げ信頼度高。相手探し（2・3号艇の差し・粘り）が主軸。"
    
    elif len(strong_outs) >= 2 and in_rate < 50.0:
        danger_boats = ", ".join([str(d["boat_no"]) for d in strong_outs[:2]])
        return f"【🔥大荒れ・外マイ警戒】 センター・外枠（**{danger_boats}号艇**）が活発。激しい攻防戦に注意。"
    
    else:
        return "【⚡差し・まくり交錯】 互角のメンバー構成。第1ターンマークの攻防に注目。"

class VenueSelect(discord.ui.Select):
    def __init__(self):
        options = [discord.SelectOption(label=v, description=f"{v}場の出走表・選手データ・展開予測を表示") for v in VENUES]
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
        
        entries_path = f"data/programs/race_cards/{year}/{month}/{day}.csv"
        df_entries = fetch_github_csv(entries_path)
        
        all_rates = load_all_course_win_rates()
        venue_rates = all_rates.get(venue, {c: 0.0 for c in range(1, 7)})
        in_rate = venue_rates.get(1, 50.0)
        tendency = VENUE_TENDENCIES.get(venue, "標準水面")
        
        summary_text = f"🏟️ **【{venue}場】 当日出走表AIスクリーニング ({date_str})**\n"
        summary_text += f"📝 水面特性: *{tendency}* (1コース勝率: **{in_rate:.1f}%**)\n\n"
        
        summary_text += "🎯 **【直近のコース別被弾・敗北傾向】**\n"
        summary_text += f"{analyze_vulnerability_trends(year, month, day)}\n\n"
        
        venue_entries = pd.DataFrame()
        if df_entries is not None and not df_entries.empty:
            col_venue = "レース場コード" if "レース場コード" in df_entries.columns else "場コード"
            if col_venue in df_entries.columns:
                venue_entries = df_entries[df_entries[col_venue].astype(str).str.zfill(2) == str(venue_code)]
        
        summary_text += "📋 **【レース別展開予測 ＆ 注目コース解説】**\n"
        for r in range(1, 13):
            racer_evals = []
            racer_structs = []
            
            if not venue_entries.empty:
                col_race = "レース回" if "レース回" in venue_entries.columns else "レース"
                if col_race in venue_entries.columns:
                    target_r_str = f"{r}R"
                    df_race = venue_entries[venue_entries[col_race].astype(str).str.contains(target_r_str)]
                    
                    if not df_race.empty:
                        row = df_race.iloc[0]
                        for b_no in range(1, 7):
                            r_name = str(row.get(f"艇{b_no}_選手名", "選手"))
                            r_class = str(row.get(f"艇{b_no}_期別", "B1"))
                            m_no = str(row.get(f"艇{b_no}_モーター番号", "-"))
                            
                            try:
                                st_val = float(row.get(f"艇{b_no}_平均ST", 0.15))
                                st_str = f"{st_val:.2f}"
                            except:
                                st_val = 0.15
                                st_str = "0.15"
                                
                            try:
                                f_int = int(row.get(f"艇{b_no}_F", 0))
                            except:
                                f_int = 0
                            f_str = f" ⚠️F{f_int}" if f_int > 0 else ""
                            
                            score = calculate_historical_motor_score(year, month, day, venue_code, m_no, days_back=60)
                            rank_str = get_short_rank(score)
                            
                            racer_structs.append({
                                "boat_no": str(b_no),
                                "class": r_class,
                                "st": st_val,
                                "f_count": f_int,
                                "score": score
                            })
                            
                            racer_evals.append(f"{b_no} {r_name}({r_class}) [ST:{st_str}{f_str}] M#{m_no}:{rank_str}")
            
            tag = generate_race_tactical_advice(racer_structs, in_rate)
                
            if racer_evals:
                evals_str = "\n   ".join(racer_evals)
                summary_text += f"・ **R{r:2d}** ➔ {tag}\n   {evals_str}\n"
            else:
                summary_text += f"・ **R{r:2d}** ➔ {tag}\n"
            
        if df_entries is None or df_entries.empty:
            summary_text += f"\n⚠️ *注意: 当日の出走表ファイル ({entries_path}) がまだ取得できないため、統計ベースの予測を表示しています。*"

        await interaction.followup.send(content=summary_text, ephemeral=True)

class VenueSelectView(discord.ui.View):
    def __init__(self):
        super().__init__()
        self.add_item(VenueSelect())

@tasks.loop(time=time(hour=8, minute=30, tzinfo=JST))
async def daily_morning_report():
    channel = bot.get_channel(NOTIFICATION_CHANNEL_ID)
    if channel is not None:
        header = "🏁 **【毎朝の自動AIスクリーニング速報（まくり差し対応・完全版）】** 🏁\nGitHubのデータ更新完了！下のメニューから気になる会場を選んで詳細をチェックしてな👇"
        await channel.send(header, view=VenueSelectView())

@daily_morning_report.before_loop
async def before_daily_report():
    await bot.wait_until_ready()

@bot.event
async def on_ready():
    print(f"Logged in as {bot.user.name}! Morning auto-report scheduler is active.")
    if not daily_morning_report.is_running():
        daily_morning_report.start()

@bot.command(name="boat_report")
async def boat_report(ctx):
    header = "🏁 **【全場AIスクリーニング速報（まくり差し対応・完全版）】** 🏁\n下のメニューから会場を選んで詳細をチェック👇"
    await ctx.send(header, view=VenueSelectView())

if __name__ == "__main__":
    token = os.environ.get("DISCORD_TOKEN")
    bot.run(token)


