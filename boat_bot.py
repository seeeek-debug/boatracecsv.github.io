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
NOTIFICATION_CHANNEL_ID = 1546632999624511610

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
        res = requests.get(url, timeout=10)
        if res.status_code == 200:
            df = pd.read_csv(io.StringIO(res.text))
            CSV_CACHE[file_path] = df
            return df
    except Exception as e:
        print(f"CSV Fetch Error ({file_path}): {e}")
    return None

# --- モデルと集計データの読み込み ---
MODEL_FILENAME = "boatrace_lgb_model.pkl"
loaded_package = None
models = None
player_course_stats = None
venue_wind_kimarite = None
player_fav_kimarite = None
player_col_name = "選手名"

try:
    loaded_package = joblib.load(MODEL_FILENAME)
    if isinstance(loaded_package, dict) and "models" in loaded_package:
        models = loaded_package["models"]
        player_course_stats = loaded_package.get("player_course_stats")
        venue_wind_kimarite = loaded_package.get("venue_wind_kimarite")
        player_fav_kimarite = loaded_package.get("player_fav_kimarite")
        player_col_name = loaded_package.get("player_col", "選手名")
        print("モデルと集計データの読み込みに成功しました。")
    else:
        # 互換性のため、従来通りモデル単体だった場合のフォールバック
        models = loaded_package
        print("モデル単体として読み込みました（集計データなし）。")
        
    if models and "rank_1" in models:
        expected_features = models["rank_1"].feature_name()
        print(f"--- モデル特徴量数: {len(expected_features)} ---")
except Exception as e:
    models = None
    print(f"モデルの読み込みに失敗しました: {e}")

def calculate_single_race_analysis(venue, venue_code, year, month, day_str, r_num):
    summary_text = f"🎯 **{venue}場 {r_num}R** のAIレース分析・展開予想 ({day_str})\n"

    if models is None or "rank_1" not in models:
        return summary_text + " ⚠️ エラー: 予測モデルが読み込まれていません。"

    expected_features = models["rank_1"].feature_name()
    day_part = day_str.split('-')[2] if '-' in day_str else day_str

    # CSV取得
    df_cards = fetch_github_csv(f"data/programs/race_cards/{year}/{month}/{day_part}.csv")
    df_sui = fetch_github_csv(f"data/previews/sui/{year}/{month}/{day_part}.csv")
    df_orig = fetch_github_csv(f"data/previews/original_exhibition/{year}/{month}/{day_part}.csv")

    if df_cards is None:
        return summary_text + " ⚠️ 注意: 出走表データが取得できませんでした。"

    def extract_race_rows(df, venue_c, r_n):
        """1レース分（6艇分）のデータを抽出する"""
        if df is None or len(df) == 0: return pd.DataFrame()
        try:
            r_str = str(r_n).zfill(2)
            venue_str = str(venue_c).zfill(2)
            target_code = f"{year}{month}{day_part}{venue_str}{r_str}"
            
            if 'レースコード' in df.columns:
                matched = df[df['レースコード'].astype(str) == target_code]
                if len(matched) > 0:
                    return matched

            venue_col = 'レース場コード' if 'レース場コード' in df.columns else ('レース場' if 'レース場' in df.columns else None)
            r_col = 'レース回' if 'レース回' in df.columns else ('R' if 'R' in df.columns else None)
            if venue_col and r_col:
                r_variants = [str(r_n), r_str, f"{r_n}R", f"{r_str}R"]
                matched = df[
                    (df[venue_col].astype(str).str.zfill(2) == venue_str) & 
                    (df[r_col].astype(str).isin(r_variants))
                ]
                if len(matched) > 0:
                    return matched
        except Exception as e:
            print(f"抽出エラー: {e}")
        return pd.DataFrame()

    df_card_rows = extract_race_rows(df_cards, venue_code, r_num)
    if len(df_card_rows) == 0:
        return summary_text + " ⚠️ 注意: 対象レースの出走表データが見つかりません。"

    df_sui_rows = extract_race_rows(df_sui, venue_code, r_num)
    df_orig_rows = extract_race_rows(df_orig, venue_code, r_num)

    # 6艇分を結合（行単位でマージするイメージ）
    # ここでは簡易的に出走表ベースに展示データを結合する
    combined_rows = []
    for idx, card_row in df_card_rows.iterrows():
        row_dict = card_row.to_dict()
        # 艇番や選手名でマッチングして展示データを補う
        boat_num = row_dict.get('枠番', row_dict.get('艇番', idx + 1))
        
        # suiやorigから該当艇のデータを探す
        def find_matching_subrow(sub_df, b_num):
            if sub_df is None or len(sub_df) == 0: return {}
            for _, s_row in sub_df.iterrows():
                if str(s_row.get('枠番', s_row.get('艇番', ''))) == str(b_num):
                    return s_row.to_dict()
            return {}

        row_dict.update(find_matching_subrow(df_sui_rows, boat_num))
        row_dict.update(find_matching_subrow(df_orig_rows, boat_num))
        combined_rows.append(row_dict)

    df_input = pd.DataFrame(combined_rows)

    # --- 級別の数値化 ---
    if "級別" in df_input.columns:
        rank_map = {'A1': 4, 'A2': 3, 'B1': 2, 'B2': 1}
        df_input["級別"] = df_input["級別"].map(rank_map).fillna(2)

    # --- 学習時と同じ集計データをマージ ---
    if player_course_stats is not None and player_col_name in df_input.columns and "枠番" in df_input.columns:
        df_input = pd.merge(df_input, player_course_stats, on=[player_col_name, "枠番"], how="left")
    
    if venue_wind_kimarite is not None and "レース場" in df_input.columns and "風向" in df_input.columns:
        df_input = pd.merge(df_input, venue_wind_kimarite, on=["レース場", "風向"], how="left")

    if player_fav_kimarite is not None and player_col_name in df_input.columns:
        df_input = pd.merge(df_input, player_fav_kimarite, on=[player_col_name], how="left")

    # 数値化処理
    for col in df_input.columns:
        if not df_input[col].dtype.name.startswith('cat') and col != player_col_name:
            df_input[col] = pd.to_numeric(df_input[col], errors='coerce')

    X_input = df_input.reindex(columns=expected_features, fill_value=0.0)

    # --- 【デバッグ確認】 ---
    debug_dict = X_input.to_dict(orient='records')[0] if len(X_input) >  0 else {}
    summary_text += f"\n🔍 **[DEBUG 特徴量確認]**\n```json\n{str(debug_dict)[:900]}\n```\n"

    # 推論実行（6艇分をまとめて予測、または1艇ずつ）
    prob_matrix = {}
    for rank_idx, rank_name in enumerate(["rank_1", "rank_2", "rank_3"]):
        if rank_name in models:
            pred_val = models[rank_name].predict(X_input)
            prob_matrix[rank_idx + 1] = np.ravel(pred_val)

    boat_data = []
    for i in range(len(df_input)):
        boat_num = i + 1
        row_d = df_input.iloc[i]
        name = f"艇{boat_num}"
        for col in row_d.keys():
            if "選手名" in col or "氏名" in col:
                if pd.notna(row_d[col]):
                    name = str(row_d[col])
                break

        arr_1 = prob_matrix.get(1, np.zeros(6))
        arr_2 = prob_matrix.get(2, np.zeros(6))
        arr_3 = prob_matrix.get(3, np.zeros(6))

        p1 = float(arr_1[i]) * 100 if len(arr_1) > i else 0.0
        p2 = float(arr_2[i]) * 100 if len(arr_2) > i else 0.0
        p3 = float(arr_3[i]) * 100 if len(arr_3) > i else 0.0

        boat_data.append({"boat": boat_num, "name": name, "p1": p1, "p2": p2, "p3": p3})

    # 表示用テキスト生成（従来通り）
    top_1st = max(boat_data, key=lambda x: x['p1']) if boat_data else {"boat": 1, "p1": 0}
    top_2nd = max(boat_data, key=lambda x: x['p2']) if boat_data else {"boat": 2, "p2": 0}

    summary_text += f"\n--- 【各艇の着順確率一覧】 ---\n"
    for bd in boat_data:
        summary_text += f"• **{bd['boat']}号艇** {bd['name']} -> 1着: **{bd['p1']:.1f}%** | 2着: **{bd['p2']:.1f}%**\n"

    summary_text += f"\n--- 【3連単 予想買い目 (上位5点)】 ---\n"
    trifecta_scores = []
    if 1 in prob_matrix and 2 in prob_matrix and 3 in prob_matrix:
        m1, m2, m3 = prob_matrix[1], prob_matrix[2], prob_matrix[3]
        for c1, c2, c3 in itertools.permutations(range(len(boat_data)), 3):
            score = float(m1[c1]) * float(m2[c2]) * float(m3[c3])
            trifecta_scores.append(((c1+1, c2+1, c3+1), score))
        trifecta_scores.sort(key=lambda x: x[1], reverse=True)

        for rank, (combo, score) in enumerate(trifecta_scores[:5], 1):
            summary_text += f"{rank}. **{combo[0]} - {combo[1]} - {combo[2]}** (スコア: {score:.4f})\n"

    return summary_text

# --- Discord UI の部分はそのまま ---
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
            traceback.print_exc()
            await interaction.followup.send(content=f"⚠️ エラーが発生しました: {e}", ephemeral=True)

class RaceSelectView(discord.ui.View):
    def __init__(self, venue):
        super().__init__(timeout=None)
        self.add_item(RaceSelect(venue))

class VenueSelect(discord.ui.Select):
    def __init__(self):
        options = [discord.SelectOption(label=v, description=f"{v} のレースを選択") for v in VENUES]
        super().__init__(
            placeholder="🏟️ 会場を選択してください...", 
            min_values=1, 
            max_values=1, 
            options=options,
            custom_id="persistent_venue_select"
        )

    async def callback(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)
        venue = self.values[0]
        await interaction.followup.send(
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

