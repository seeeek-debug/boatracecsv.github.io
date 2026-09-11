import asyncio
from datetime import datetime, time, timezone, timedelta
import io
import os
import threading
import traceback
from flask import Flask
import discord
from discord.ext import commands, tasks
import numpy as np
import pandas as pd
import requests
import joblib
import itertools

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
NOTIFICATION_CHANNEL_ID = 1546632999624511610  # ←必要に応じて変更

JST = timezone(timedelta(hours=9))

VENUES = [
    "桐生", "戸田", "江戸川", "平和島", "多摩川", "浜名湖",
    "蒲郡", "常滑", "津", "三国", "びわこ", "住之江",
    "尼崎", "鳴門", "丸亀", "児島", "宮島", "徳山",
    "下関", "若松", "芦屋", "福岡", "唐津", "大村"
]

VENUE_MAPPING = {
    "桐生": "01", "戸田": "02", "江戸川": "03", "平和島": "04", "多摩川": "05", "浜名湖": "06",
    "蒲郡": "07", "常滑": "08", "津": "09", "三国": "10", "びわこ": "11", "住之江": "12",
    "尼崎": "13", "鳴門": "14", "丸亀": "15", "児島": "16", "宮島": "17", "徳山": "18",
    "下関": "19", "若松": "20", "芦屋": "21", "福岡": "22", "唐津": "23", "大村": "24"
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

# --- モデル読み込み ---
MODEL_FILENAME = "boatrace_lgb_model.pkl"
try:
    models = joblib.load(MODEL_FILENAME)
    print(f"モデル '{MODEL_FILENAME}' の読み込みに成功しました。")
except Exception as e:
    models = None
    print(f"モデルの読み込みに失敗しました: {e}")

def calculate_single_race_analysis(venue, venue_code, year, month, day_str, r_num):
    summary_text = f"🎯 **{venue}場 {r_num}R** のAIレース分析・展開予想 ({day_str})\n"

    if models is None:
        return summary_text + " ⚠️ エラー: 予測モデルが読み込まれていません。"

    if "rank_1" in models:
        expected_features = models["rank_1"].feature_name()
    else:
        return summary_text + " ⚠️ エラー: モデル内に rank_1 が見つかりません。"

    day_part = day_str.split('-')[2] if '-' in day_str else day_str
    df_cards = fetch_github_csv(f"data/programs/race_cards/{year}/{month}/{day_part}.csv")
    df_sui = fetch_github_csv(f"data/previews/sui/{year}/{month}/{day_part}.csv")
    df_orig = fetch_github_csv(f"data/previews/original_exhibition/{year}/{month}/{day_part}.csv")

    if df_cards is None:
        return summary_text + " ⚠️ 注意: 出走表データが取得できませんでした。"

    def filter_race_data(df, venue_c, r_n):
        if df is None: return None
        for col in df.columns:
            matched = df[df[col].astype(str).str.contains(str(venue_c)) & df[col].astype(str).str.contains(str(r_n))]
            if len(matched) > 0:
                return matched.iloc[0:1].copy()
        return None

    df_c_row = filter_race_data(df_cards, venue_code, r_num)
    df_sui_row = filter_race_data(df_sui, venue_code, r_num)
    df_orig_row = filter_race_data(df_orig, venue_code, r_num)

    if df_c_row is None or len(df_c_row) == 0:
        return summary_text + " ⚠️ 注意: 対象レースのデータが見つかりません。"

    # 横持ちデータ（1レース1行）を結合して構築
    combined_row = {}
    for df_r in [df_c_row, df_sui_row, df_orig_row]:
        if df_r is not None and len(df_r) > 0:
            for col in df_r.columns:
                combined_row[col] = df_r[col].values[0]

    df_input_row = pd.DataFrame([combined_row])

    player_cols = [col for col in df_input_row.columns if "選手コード" in col or "登録番号" in col]
    for col in player_cols:
        df_input_row[col] = df_input_row[col].astype('category')

    for col in df_input_row.columns:
        if col not in player_cols and not df_input_row[col].dtype.name.startswith('cat'):
            df_input_row[col] = pd.to_numeric(df_input_row[col], errors='coerce')

    X_input = df_input_row.reindex(columns=expected_features, fill_value=0.0)

    prob_matrix = {}
    for rank_idx, rank_name in enumerate(["rank_1", "rank_2", "rank_3"]):
        if rank_name in models:
            model = models[rank_name]
            pred_val = model.predict(X_input)
            prob_matrix[rank_idx + 1] = np.ravel(pred_val)

    boat_data = []
    for i in range(6):
        boat_num = i + 1
        
        name = f"艇{boat_num}"
        for col in df_c_row.columns:
            if f"艇{boat_num}" in col and "選手名" in col:
                val = df_c_row[col].values[0]
                if pd.notna(val):
                    name = str(val)
                break

        arr_1 = prob_matrix.get(1, np.zeros(6))
        arr_2 = prob_matrix.get(2, np.zeros(6))
        arr_3 = prob_matrix.get(3, np.zeros(6))

        p1 = float(arr_1[i]) * 100 if len(arr_1) > i else 0.0
        p2 = float(arr_2[i]) * 100 if len(arr_2) > i else 0.0
        p3 = float(arr_3[i]) * 100 if len(arr_3) > i else 0.0

        boat_data.append({"boat": boat_num, "name": name, "p1": p1, "p2": p2, "p3": p3})

    top_1st = max(boat_data, key=lambda x: x['p1']) if boat_data else {"boat": 1, "p1": 0}
    top_2nd = max(boat_data, key=lambda x: x['p2']) if boat_data else {"boat": 2, "p2": 0}

    # --- 展開予測の判定 ---
    if len(boat_data) >= 1 and boat_data[0]['p1'] >= 38.0:
        tactical_tag = "🛡️ 【イン鉄壁・逃げ本線】 1号艇が抜群の信頼度で逃走"
        kimarite = "逃げ (1-2, 1-3)"
    elif len(boat_data) >= 2 and boat_data[1]['p1'] >= 20.0 and boat_data[1]['p1'] > boat_data[0]['p1']:
        tactical_tag = "💡 【2号艇の差し抜け】 2コースから鋭く差し込む"
        kimarite = "差し (2-1, 2-3)"
    elif len(boat_data) >= 2 and boat_data[1]['p1'] >= 23.0:
        tactical_tag = "⚡ 【2号艇まくり展開】 伸び足を活かしてインを襲う"
        kimarite = "まくり (2-3, 2-4)"
    elif len(boat_data) >= 3 and boat_data[2]['p1'] >= 18.0:
        tactical_tag = "🌊 【3号艇のセンター強襲】 自在に攻めて主導権を握る"
        kimarite = "まくり差し / まくり (3-1, 3-2)"
    elif len(boat_data) >= 6 and (boat_data[5]['p1'] >= 15.0 or boat_data[4]['p1'] >= 12.0 or boat_data[3]['p1'] >= 10.0):
        out_candidates = boat_data[3:]
        best_out = max(out_candidates, key=lambda x: x['p1'])
        if best_out['boat'] == 4:
            tactical_tag = "🔥 【4号艇のカド一撃・まくり展開】 助走の踏み込みから絞りマイの展開を作る"
            kimarite = "まくり / まくり差し (4-1, 4-5)"
        else:
            tactical_tag = f"🎯 【{best_out['boat']}号艇の外マイ・まくり差し】 展開の隙を突く鋭い仕掛け"
            kimarite = f"まくり差し / 差し ({best_out['boat']}-1, {best_out['boat']}-2)"
    else:
        tactical_tag = "⚔️ 【混戦・差し手モツレ】 互いの攻防から手堅く潰す展開"
        kimarite = "差し / 差し継ぎ"

    summary_text += f"\n--- 【展開予想】 ---\n{tactical_tag}\n"
    summary_text += f"🎯 **推奨決まり手**: {kimarite}\n"

    summary_text += f"\n--- 【各艇の着順確率一覧】 ---\n"
    for bd in boat_data:
        summary_text += f"• **{bd['boat']}号艇** {bd['name']} -> 1着: **{bd['p1']:.1f}%** | 2着: **{bd['p2']:.1f}%**\n"

    summary_text += f"\n--- 【3連単 予想買い目 (上位5点)】 ---\n"
    trifecta_scores = []
    if 1 in prob_matrix and 2 in prob_matrix and 3 in prob_matrix:
        m1 = np.ravel(prob_matrix[1])
        m2 = np.ravel(prob_matrix[2])
        m3 = np.ravel(prob_matrix[3])

        for c1, c2, c3 in itertools.permutations(range(6), 3):
            s1 = float(m1[c1]) if len(m1) > c1 else 0.0
            s2 = float(m2[c2]) if len(m2) > c2 else 0.0
            s3 = float(m3[c3]) if len(m3) > c3 else 0.0
            score = s1 * s2 * s3
            trifecta_scores.append(((c1+1, c2+1, c3+1), score))

        trifecta_scores.sort(key=lambda x: x[1], reverse=True)

        for rank, (combo, score) in enumerate(trifecta_scores[:5], 1):
            summary_text += f"{rank}. **{combo[0]} - {combo[1]} - {combo[2]}** (スコア: {score:.4f})\n"

    summary_text += f"\n--- 【レース展開の考察】 ---\n"
    if boat_data:
        summary_text += f"• 軸推奨: **{top_1st['boat']}号艇 {top_1st['name']}** (1着トップ: {top_1st['p1']:.1f}%)\n"
        summary_text += f"• 相手候補: **{top_2nd['boat']}号艇 {top_2nd['name']}** (2番手力: {top_2nd['p2']:.1f}%)\n"

    return summary_text

# --- 競走セレクトメニューの定義 ---
class RaceSelect(discord.ui.Select):
    def __init__(self, venue):
        self.venue = venue
        options = [discord.SelectOption(label="⭐ 全レース一括予想 (1R～12R)", value="all")]
        for i in range(1, 13):
            options.append(discord.SelectOption(label=f"🎯 {i}レース ({i}R)", value=str(i)))
        super().__init__(placeholder="👇 分析するレースを選択してください...", min_values=1, max_values=1, options=options)

    async def callback(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)
        try:
            venue = self.venue
            venue_code = VENUE_MAPPING.get(venue, "01")
            val = self.values[0]

            target_date = datetime.now(JST)
            year = target_date.strftime("%Y")
            month = target_date.strftime("%m")
            day_str = target_date.strftime("%Y-%m-%d")

            if val == "all":
                all_summaries = [f"📊 **{venue}場** 全12レースAI予測・展開予想一覧\n"]
                for r_num in range(1, 13):
                    res_text = calculate_single_race_analysis(venue, venue_code, year, month, day_str, r_num)
                    all_summaries.append(res_text + "\n" + "="*30 + "\n")

                result_text = "\n".join(all_summaries)
            else:
                r_num = int(val)
                result_text = calculate_single_race_analysis(venue, venue_code, year, month, day_str, r_num)

            if len(result_text) <= 2000:
                await interaction.followup.send(content=result_text, ephemeral=True)
            else:
                for i in range(0, len(result_text), 2000):
                    await interaction.followup.send(content=result_text[i:i+2000], ephemeral=True)
        except Exception as e:
            print("--- 詳細なエラー内容 ---")
            traceback.print_exc()
            await interaction.followup.send(content=f"⚠️ エラーが発生しました: {e}", ephemeral=True)

class RaceSelectView(discord.ui.View):
    def __init__(self, venue):
        super().__init__(timeout=None)
        self.add_item(RaceSelect(venue))

class VenueSelect(discord.ui.Select):
    def __init__(self):
        options = [
            discord.SelectOption(label=v, description=f"{v} のレースを選択") for v in VENUES
        ]
        super().__init__(placeholder="🏟️ 会場を選択してください...", min_values=1, max_values=1, options=options)

    async def callback(self, interaction: discord.Interaction):
        venue = self.values[0]
        await interaction.response.send_message(
            content=f"🏟️ **{venue}場** が選択されました。続いて、予測・展開を見たいレースを選択してください。",
            view=RaceSelectView(venue),
            ephemeral=True
        )

class VenueSelectView(discord.ui.View):
    def __init__(self):
        super().__init__(timeout=None)
        self.add_item(VenueSelect())

@bot.event
async def on_ready():
    print(f'Logged in as {bot.user.name}')

    try:
        bot.add_view(VenueSelectView())
    except Exception as e:
        print(f"Error adding view: {e}")

    try:
        channel_id = int(NOTIFICATION_CHANNEL_ID)
        channel = bot.get_channel(channel_id)
        if channel is None:
            channel = await bot.fetch_channel(channel_id)

        if channel:
            await channel.send(
                content="🤖 **【AIレース分析・展開メニュー】**\n👇 下のメニューからいつでも会場を選択して予測を実行できます！",
                view=VenueSelectView()
            )
            print("初期メニューの自動送信に成功しました！")
    except Exception as e:
        print(f"自動送信エラー (無視して動作に影響はありません): {e}")

@bot.command(name="setup")
async def setup_menu(ctx):
    await ctx.message.delete()
    await ctx.send(
        content="🤖 **【AIレース分析・展開メニュー】**\n👇 下のメニューからいつでも会場を選択して予測を実行できます！",
        view=VenueSelectView()
    )

if __name__ == "__main__":
    keep_alive()
    token = os.environ.get("DISCORD_TOKEN")
    bot.run(token)

