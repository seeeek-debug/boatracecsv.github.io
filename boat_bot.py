import asyncio
from datetime import datetime, timezone, timedelta
import io
import os
import threading
import traceback
from flask import Flask
import discord
from discord.ext import commands
import numpy as np
import pandas as pd
import requests
import joblib
import itertools

# --- Render用Webサーバー ---
app = Flask(__name__)

@app.route("/")
def home():
    return "I am alive"

def run_web():
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, use_reloader=False)

def keep_alive():
    t = threading.Thread(target=run_web)
    t.daemon = True
    t.start()

# --- Discordボット設定 ---
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/seeeek-debug/boatracecsv.github.io/main/"
NOTIFICATION_CHANNEL_ID = 156632996264511610

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
    uri = f"{GITHUB_RAW_BASE}{file_path}"
    try:
        res = requests.get(uri, timeout=10)
        if res.status_code == 200:
            df = pd.read_csv(io.StringIO(res.text), dtype=str)
            df.columns = df.columns.str.strip()
            CSV_CACHE[file_path] = df
            return df
    except Exception as e:
        print(f"CSV Fetch Error ({file_path}): {e}")
    return None

# --- モデルの読み込み ---
MODEL_FILENAME = "boatrace_lgb_model.pkl"
loaded_package = None
models = None

try:
    loaded_package = joblib.load(MODEL_FILENAME)
    if isinstance(loaded_package, dict) and "models" in loaded_package:
        models = loaded_package["models"]
        print("モデルの読み込みに成功しました。")
    else:
        models = loaded_package
        print("モデル単体として読み込みました。")
        
    if models and "rank_1" in models:
        expected_features = models["rank_1"].feature_name()
        print(f"モデル特徴量数: {len(expected_features)}")
except Exception as e:
    models = None
    print(f"モデルの読み込みに失敗しました: {e}")

def calculate_single_race_analysis(venue, venue_code, year, month, day_str, r_num):
    summary_text = f"🤖 **{venue}** {r_num}RのAIレース分析・局面予想 ({day_str})\n"

    if models is None or "rank_1" not in models:
        return summary_text + " ⚠️ エラー: 予測モデルが読み込まれていません。"

    expected_features = models["rank_1"].feature_name()
    day_part = day_str.split("-")[2] if "-" in day_str else day_str
    
    race_card_path = f"data/programs/race_cards/{year}/{month}/{day_part}.csv"
    sui_path = f"data/previews/sui/{year}/{month}/{day_part}.csv"
    orig_path = f"data/previews/original_exhibition/{year}/{month}/{day_part}.csv"
    
    df_cards = fetch_github_csv(race_card_path)
    df_sui = fetch_github_csv(sui_path)
    df_orig = fetch_github_csv(orig_path)

    if df_cards is None:
        return summary_text + " ⚠️ エラー: 出走表データが取得できませんでした。"

    r_str = str(r_num).zfill(2)
    venue_s = str(venue_code).zfill(2)
    target_race_code = f"{year}{month}{day_part}{venue_s}{r_str}"

    def get_matched_row(df, code):
        if df is None: return None
        for col in df.columns:
            clean_col = df[col].astype(str).str.strip().str.split('.').str[0].str.lstrip("0")
            clean_target = str(code).strip().split('.')[0].lstrip("0")
            matched = df[clean_col == clean_target]
            if len(matched) > 0:
                return matched.iloc[0:1].copy()
        return None

    df_c_row = get_matched_row(df_cards, target_race_code)
    if df_c_row is None or len(df_c_row) == 0:
        return summary_text + f" ⚠️ エラー: レースコード '{target_race_code}' が出走表に見つかりませんでした。"

    df_s_row = get_matched_row(df_sui, target_race_code)
    df_o_row = get_matched_row(df_orig, target_race_code)

    base_info = {}
    for col in df_c_row.columns:
        if not col.startswith("艇"):
            base_info[col] = df_c_row[col].values[0]

    for df_r in [df_s_row, df_o_row]:
        if df_r is not None:
            for c in df_r.columns:
                if not c.startswith("艇"):
                    base_info[c] = df_r[c].values[0]

    vertical_rows = []
    for i in range(1, 7):
        row_data = base_info.copy()
        row_data["枠番"] = i
        for df_r in [df_c_row, df_s_row, df_o_row]:
            if df_r is not None:
                for col in df_r.columns:
                    if col.startswith(f"艇{i}_"):
                        row_data[col.replace(f"艇{i}_", "")] = df_r[col].values[0]
        vertical_rows.append(row_data)

    df_target = pd.DataFrame(vertical_rows)

    if "級別" in df_target.columns:
        rank_map = {'A1': 4, 'A2': 3, 'B1': 2, 'B2': 1}
        df_target["級別"] = df_target["級別"].map(rank_map)

    player_col = next((col for col in ["選手コード", "登録番号"] if col in df_target.columns), None)
    if player_col:
        df_target[player_col] = df_target[player_col].astype('category').cat.codes

    for col in df_target.columns:
        if col not in [player_col, "選手名", "支部", "出身地"]:
            df_target[col] = pd.to_numeric(df_target[col], errors='coerce')

    X_input = df_target.reindex(columns=expected_features, fill_value=0.0)

    prob_matrix = {}
    for rank_idx, rank_name in enumerate(["rank_1", "rank_2", "rank_3"], 1):
        if rank_name in models:
            model = models[rank_name]
            preds_per_boat = []
            for idx, row in X_input.iterrows():
                p = model.predict(row.values.reshape(1, -1))[0]
                preds_per_boat.append(p)
            prob_matrix[rank_idx] = np.array(preds_per_boat)

    boat_data = []
    summary_text += f"\n--- 【各艇の着順確率一覧】 ---\n"
    for i in range(6):
        boat_num = i + 1
        name = str(df_target.loc[i, "選手名"]) if "選手名" in df_target.columns and pd.notna(df_target.loc[i, "選手名"]) else f"選手{boat_num}"
        
        # モデルの出力に応じた確率取得
        arr_1 = prob_matrix.get(1, np.zeros(6))
        arr_2 = prob_matrix.get(2, np.zeros(6))
        arr_3 = prob_matrix.get(3, np.zeros(6))
        
        p1 = float(arr_1[i]) * 100 if len(arr_1) > i else 0.0
        p2 = float(arr_2[i]) * 100 if len(arr_2) > i else 0.0
        p3 = float(arr_3[i]) * 100 if len(arr_3) > i else 0.0
        
        boat_data.append({"boat": boat_num, "name": name, "p1": p1, "p2": p2, "p3": p3})
        summary_text += f"• **{boat_num}号艇** {name} -> 1着: **{p1:.1f}%** | 2着: **{p2:.1f}%**\n"

    if not boat_data:
        return summary_text + " ⚠️ エラー: 艇データが取得できませんでした。"

    top_1st = max(boat_data, key=lambda x: x['p1'])
    top_2nd = max(boat_data, key=lambda x: x['p2'])

    if top_1st['boat'] == 1 and top_1st['p1'] >= 40.0:
        tactical_tag = "🛡️ 【イン鉄壁・逃げ本線】 1号艇が抜群の信頼度で逃走"
        kimarite = "逃げ (1-2, 1-3)"
    elif top_1st['boat'] == 2:
        tactical_tag = "💡 【2号艇の差し抜け】 2コースから鋭く差し込む"
        kimarite = "差し (2-1, 2-3)"
    elif top_1st['boat'] == 3:
        tactical_tag = "🌊 【3号艇のセンター強襲】 自在に攻めて主導権を握る"
        kimarite = "まくり差し / まくり (3-1, 3-2)"
    elif top_1st['boat'] >= 4:
        tactical_tag = f"🔥 【{top_1st['boat']}号艇の展開突き・強襲】 外枠から一撃を狙う"
        kimarite = f"まくり差し / 差し ({top_1st['boat']}-1, {top_1st['boat']}-2)"
    else:
        tactical_tag = "⚔️ 【混戦・差し手モツレ】 互いの攻防から手堅く潰す展開"
        kimarite = "差し / 差し継ぎ"

    summary_text = f"🤖 **{venue}** {r_num}RのAIレース分析・局面予想 ({day_str})\n" \
                   f"\n--- 【展開予想】 ---\n{tactical_tag}\n" \
                   f"🎯 **推奨決まり手**: {kimarite}\n" + summary_text[summary_text.find("--- 【各艇の着順"): ]

    summary_text += f"\n--- 【3連単 予想買い目 (上位5点)】 ---\n"
    trifecta_scores = []
    if 1 in prob_matrix and 2 in prob_matrix and 3 in prob_matrix:
        m1, m2, m3 = prob_matrix[1], prob_matrix[2], prob_matrix[3]
        for c1, c2, c3 in itertools.permutations(range(6), 3):
            score = float(m1[c1]) * float(m2[c2]) * float(m3[c3])
            trifecta_scores.append(((c1+1, c2+1, c3+1), score))
        trifecta_scores.sort(key=lambda x: x[1], reverse=True)

        for rank, (combo, score) in enumerate(trifecta_scores[:5], 1):
            summary_text += f"{rank}. **{combo[0]} - {combo[1]} - {combo[2]}** (スコア: {score:.4f})\n"

    summary_text += f"\n--- 【レース展開の考察】 ---\n"
    summary_text += f"• 軸推奨: **{top_1st['boat']}号艇 {top_1st['name']}** (1着トップ: {top_1st['p1']:.1f}%)\n"
    summary_text += f"• 相手候補: **{top_2nd['boat']}号艇 {top_2nd['name']}** (2番手力: {top_2nd['p2']:.1f}%)\n"

    return summary_text

# --- Discord UI部分 ---
class RaceSelect(discord.ui.Select):
    def __init__(self, venue):
        self.venue = venue
        options = [discord.SelectOption(label="⭐ 全レース予想 (1R～12R)", value="all")]
        for i in range(1, 13):
            options.append(discord.SelectOption(label=f"💖 {i}レース ({i}R)", value=str(i)))
        super().__init__(placeholder="分析するレースを選択してください...", min_values=1, max_values=1, options=options)

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
                all_summaries = [f"🤖 **{venue}** 全12レースAI予測・展開予想一覧"]
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
            traceback.print_exc()
            await interaction.followup.send(content=f"⚠️ エラーが発生しました: {e}", ephemeral=True)

class RaceSelectView(discord.ui.View):
    def __init__(self, venue):
        super().__init__(timeout=None)
        self.add_item(RaceSelect(venue))

class VenueSelect(discord.ui.Select):
    def __init__(self):
        options = [discord.SelectOption(label=v, description=f"{v} のレースを選択") for v in VENUES]
        super().__init__(placeholder="会場を選択してください...", min_values=1, max_values=1, options=options, custom_id="persistent_venue_select")

    async def callback(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)
        venue = self.values[0]
        await interaction.followup.send(
            content=f"🏟️ **{venue}** が選択されました。続いて、予測・展開を見たいレースを選択してください。",
            view=RaceSelectView(venue),
            ephemeral=True
        )

class VenueSelectView(discord.ui.View):
    def __init__(self):
        super().__init__(timeout=None)
        self.add_item(VenueSelect())

@bot.event
async def on_ready():
    print(f"Logged in as {bot.user.name}")
    try:
        bot.add_view(VenueSelectView())
    except Exception as e:
        print(f"Error adding view: {e}")

@bot.command(name="setup")
async def setup(ctx):
    await ctx.message.delete()
    await ctx.send(
        content="🤖 **【AIレース分析・展開メニュー】**\n👇 下のメニューからいつでも会場を選択して予測を実行できます！",
        view=VenueSelectView()
    )

if __name__ == "__main__":
    keep_alive()
    token = os.environ.get("DISCORD_TOKEN")
    bot.run(token)

