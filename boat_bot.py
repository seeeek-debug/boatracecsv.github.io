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

VENUE_PREVIEW_CODE_MAP = {
    "01": "kir", "02": "tod", "03": "edg", "04": "hei", "05": "tam", "06": "ham",
    "07": "gam", "08": "tkz", "09": "tsu", "10": "mik", "11": "biw", "12": "sum",
    "13": "ama", "14": "nar", "15": "mar", "16": "koj", "17": "miy", "18": "tok",
    "19": "shm", "20": "wkm", "21": "ash", "22": "fuk", "23": "ktu", "24": "omr"
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
            df = pd.read_csv(io.StringIO(res.text), encoding="utf-8-sig", dtype=str)
            df.columns = df.columns.str.strip()
            CSV_CACHE[file_path] = df
            return df
    except Exception as e:
        print(f"CSV Fetch Error ({file_path}): {e}")
    return None

# --- モデルおよびデータの読み込み ---
MODEL_FILENAME = "boatrace_lgb_model.pkl"
loaded_package = None
models = None
player_fav_kimarite = None
kimarite_prob_dict = {}
expected_features = []

try:
    loaded_package = joblib.load(MODEL_FILENAME)
    if isinstance(loaded_package, dict):
        models = loaded_package.get("models")
        player_fav_kimarite = loaded_package.get("player_fav_kimarite")
        loaded_pair_table = loaded_package.get("pair_table")
        if loaded_pair_table and isinstance(loaded_pair_table, dict):
            kimarite_prob_dict = loaded_pair_table
        print("パッケージ形式でモデルとデータを読み込みました。")
    else:
        models = loaded_package
        print("モデル単体として読み込みました。")
        
    if models and "rank_1" in models:
        expected_features = models["rank_1"].feature_name()
        print(f"モデル特徴量数: {len(expected_features)}")
except Exception as e:
    models = None
    print(f"モデルの読み込みに失敗しました: {e}")

def load_kimarite_table_from_github():
    global kimarite_prob_dict
    if kimarite_prob_dict:
        return
    df_pair = fetch_github_csv("pair_table.csv")
    if df_pair is not None:
        try:
            for _, row in df_pair.iterrows():
                k_type = str(row['セル']).strip()
                c2 = int(row['2着コース'])
                c3 = int(row['3着コース'])
                prob = float(row['確率'])
                kimarite_prob_dict[(k_type, c2, c3)] = prob
        except Exception as e:
            print(f"pair_table.csv パースエラー: {e}")

load_kimarite_table_from_github()

def get_season(m):
    if m in [3, 4, 5]: return "春"
    elif m in [6, 7, 8]: return "夏"
    elif m in [9, 10, 11]: return "秋"
    else: return "冬"

def calculate_single_race_analysis(venue, venue_code, year, month, day_str, r_num):
    summary_text = f"🤖 **{venue}** {r_num}RのAIレース分析・局面予想 ({day_str})\n"

    if models is None or "rank_1" not in models:
        return summary_text + " ⚠️ エラー: 予測モデルが読み込まれていません。"

    day_part = day_str.split("-")[2] if "-" in day_str else day_str
    venue_s = str(venue_code).zfill(2)
    month_int = int(month)
    
    race_card_path = f"data/programs/race_cards/{year}/{month}/{day_part}.csv"
    sui_path = f"data/previews/sui/{year}/{month}/{day_part}.csv"
    orig_path = f"data/previews/original_exhibition/{year}/{month}/{day_part}.csv"
    stt_path = f"data/previews/stt/{year}/{month}/{day_part}.csv"
    
    prev_code = VENUE_PREVIEW_CODE_MAP.get(venue_s, "")
    venue_preview_path = f"data/previews/{prev_code}/{year}/{month}/{day_part}.csv" if prev_code else ""

    df_cards = fetch_github_csv(race_card_path)
    df_sui = fetch_github_csv(sui_path)
    df_orig = fetch_github_csv(orig_path)
    df_stt = fetch_github_csv(stt_path)
    df_venue_preview = fetch_github_csv(venue_preview_path) if venue_preview_path else None

    df_course_win = fetch_github_csv("data/estimate/stadium/course_win_rate.csv")
    df_season_win = fetch_github_csv("data/estimate/stadium/win_rate.csv")

    if df_cards is None:
        return summary_text + f" ⚠️ エラー: 出走表データが取得できませんでした ({race_card_path})。"

    r_str = str(r_num).zfill(2)
    target_race_code = f"{year}{month}{day_part}{venue_s}{r_str}"

    def get_matched_row(df, code):
        if df is None: return None
        for col in df.columns:
            if "レースコード" in col or "code" in col.lower():
                matched = df[df[col].astype(str).str.strip() == str(code)]
                if len(matched) > 0:
                    return matched.iloc[0].to_dict()
        return None

    card_row = get_matched_row(df_cards, target_race_code)
    if not card_row:
        return summary_text + f" ⚠️ エラー: レースコード '{target_race_code}' が出走表に見つかりませんでした。"

    sui_row = get_matched_row(df_sui, target_race_code) or {}
    orig_row = get_matched_row(df_orig, target_race_code) or {}
    stt_row = get_matched_row(df_stt, target_race_code) or {}
    venue_preview_row = get_matched_row(df_venue_preview, target_race_code) or {}

    combined_row = {}
    combined_row.update(card_row)
    combined_row.update(sui_row)
    combined_row.update(orig_row)
    combined_row.update(stt_row)
    combined_row.update(venue_preview_row)

    if df_course_win is not None:
        for _, row in df_course_win.iterrows():
            v_code = str(row.get("場コード", "")).strip().zfill(2)
            r_num_str = str(row.get("レース回", "")).strip()
            if v_code == venue_s and r_num_str == str(int(r_num)):
                for k, v in row.items():
                    if k not in ["場コード", "レース回"]:
                        combined_row[f"est_course_{k}"] = v
                break

    if df_season_win is not None:
        season_name = get_season(month_int)
        for _, row in df_season_win.iterrows():
            v_code = str(row.get("場コード", "")).strip().zfill(2)
            season = str(row.get("季節", "")).strip()
            if v_code == venue_s and season == season_name:
                for k, v in row.items():
                    if k not in ["場コード", "季節"]:
                        combined_row[f"est_season_{k}"] = v
                break

    df_pred = pd.DataFrame([combined_row])

    if player_fav_kimarite:
        for i in range(1, 7):
            p_col_candidates = [f"艇{i}_選手名", f"{i}号艇_選手名", f"選手名_{i}", f"選手{i}_名前"]
            p_col = next((c for c in p_col_candidates if c in df_pred.columns), None)
            
            if p_col:
                dummy_k_keys = list(next(iter(player_fav_kimarite.values())).keys()) if player_fav_kimarite else []
                for k_name in dummy_k_keys:
                    col_name = f"艇{i}_kimarite_{k_name}"
                    p_val = str(df_pred.iloc[0].get(p_col, "")).strip()
                    df_pred[col_name] = player_fav_kimarite.get(p_val, {}).get(k_name, 0.0)

    feature_cols = [col for col in df_pred.columns if not col.startswith("res_")]
    
    for col in feature_cols:
        if col not in ["レース場", "風向", "天候"]:
            df_pred[col] = pd.to_numeric(df_pred[col], errors='coerce')

    for col in ["レース場", "風向", "天候"]:
        if col in df_pred.columns:
            df_pred[col] = df_pred[col].astype('category')

    X_input = df_pred.reindex(columns=expected_features, fill_value=0.0)
    for col in expected_features:
        if col in ["レース場", "風向", "天候"] and col in X_input.columns:
            X_input[col] = X_input[col].astype('category')

    for col in X_input.select_dtypes(include=[np.number]).columns:
        X_input[col] = X_input[col].fillna(0.0)

    prob_matrix = {}
    for rank_idx, rank_name in enumerate(["rank_1", "rank_2", "rank_3"], 1):
        if rank_name in models:
            model = models[rank_name]
            preds = model.predict(X_input)
            if len(preds) > 0:
                prob_matrix[rank_idx] = np.array(preds[0])

    boat_data = []
    summary_text += f"\n--- 【各艇の着順確率一覧】 ---\n"
    
    for i in range(6):
        boat_num = i + 1
        
        name = f"選手{boat_num}"
        for c in [f"艇{boat_num}_選手名", f"{boat_num}号艇_選手名", f"選手{boat_num}_名前"]:
            if c in card_row and pd.notna(card_row[c]):
                val = str(card_row[c]).strip()
                if val and val != "nan":
                    name = val
                    break

        p_class = ""
        for c in [f"艇{boat_num}_級別", f"{boat_num}号艇_級別", f"選手{boat_num}_級別", f"艇{boat_num}_級"]:
            if c in card_row and pd.notna(card_row[c]):
                val = str(card_row[c]).strip()
                if val and val != "nan":
                    p_class = val
                    break
        
        arr_1 = prob_matrix.get(1, np.zeros(6))
        arr_2 = prob_matrix.get(2, np.zeros(6))
        arr_3 = prob_matrix.get(3, np.zeros(6))
        
        p1 = float(arr_1[i]) * 100 if len(arr_1) > i else 0.0
        p2 = float(arr_2[i]) * 100 if len(arr_2) > i else 0.0
        p3 = float(arr_3[i]) * 100 if len(arr_3) > i else 0.0
        
        boat_data.append({"boat": boat_num, "name": name, "class": p_class, "p1": p1, "p2": p2, "p3": p3})
        
        class_str = f" ({p_class})" if p_class else ""
        summary_text += f"・{boat_num}号艇 {name}{class_str} -> 1着: {p1:.1f}% | 2着: {p2:.1f}% | 3着: {p3:.1f}%\n"

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
        
        entry_courses = {i+1: i+1 for i in range(6)}
        default_kimarite_map = {
            1: "逃げ", 2: "差し", 3: "まくり", 4: "まくり", 5: "まくり差し", 6: "差し"
        }

        for c1_idx, c2_idx, c3_idx in itertools.permutations(range(6), 3):
            b1 = c1_idx + 1
            b2 = c2_idx + 1
            b3 = c3_idx + 1
            
            p1 = float(m1[c1_idx])
            p2 = float(m2[c2_idx])
            p3 = float(m3[c3_idx])
            
            ai_base_score = (p1 ** 1.8) * (p2 ** 1.3) * (p3 ** 1.0)
            
            c1_course = entry_courses[b1]
            c2_course = entry_courses[b2]
            c3_course = entry_courses[b3]
            
            primary_kimarite = default_kimarite_map.get(b1, "差し")
            k_key = f"{primary_kimarite}_{c1_course}"
            pair_prob = kimarite_prob_dict.get((k_key, c2_course, c3_course), 0.01)
            
            final_score = ai_base_score * (max(pair_prob, 0.001) ** 0.3)
            trifecta_scores.append(((b1, b2, b3), final_score))
            
        trifecta_scores.sort(key=lambda x: x[1], reverse=True)

        for rank, (combo, score) in enumerate(trifecta_scores[:5], 1):
            summary_text += f"{rank}. **{combo[0]} - {combo[1]} - {combo[2]}** (スコア: {score:.4f})\n"

    top_1_class_str = f" ({top_1st['class']})" if top_1st['class'] else ""
    top_2_class_str = f" ({top_2nd['class']})" if top_2nd['class'] else ""

    summary_text += f"\n--- 【レース展開の考察】 ---\n"
    summary_text += f"• 軸推奨: **{top_1st['boat']}号艇 {top_1st['name']}{top_1_class_str}** (1着トップ: {top_1st['p1']:.1f}%)\n"
    summary_text += f"• 相手候補: **{top_2nd['boat']}号艇 {top_2nd['name']}{top_2_class_str}** (2番手力: {top_2nd['p2']:.1f}%)\n"

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
            tb = traceback.format_exc()
            error_msg = f"⚠️ エラーが発生しました:\n```python\n{tb}\n```"
            if len(error_msg) > 2000:
                error_msg = error_msg[:1990] + "\n```"
            await interaction.followup.send(content=error_msg, ephemeral=True)

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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, timeout=None)
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

