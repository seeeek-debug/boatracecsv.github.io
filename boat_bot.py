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

try:
    loaded_package = joblib.load(MODEL_FILENAME)
    if isinstance(loaded_package, dict) and "models" in loaded_package:
        models = loaded_package["models"]
        player_course_stats = loaded_package.get("player_course_stats")
        venue_wind_kimarite = loaded_package.get("venue_wind_kimarite")
        player_fav_kimarite = loaded_package.get("player_fav_kimarite")
        print("モデルと集計データの読み込みに成功しました。")
    else:
        models = loaded_package
        print("モデル単体として読み込みました（集計データなし）。")
        
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
    
    prog_path = f"data/programs/race_cards/{year}/{month}/{day_part}.csv"
    sui_path = f"data/previews/sui/{year}/{month}/{day_part}.csv"
    orig_path = f"data/previews/original_exhibition/{year}/{month}/{day_part}.csv"
    
    df_cards = fetch_github_csv(prog_path)
    df_sui = fetch_github_csv(sui_path)
    df_orig = fetch_github_csv(orig_path)
    
    r_str = str(r_num).zfill(2)
    venue_s = str(venue_code).zfill(2)
    target_code = f"{year}{month}{day_part}{venue_s}{r_str}"

    def extract_target_row(df):
        if df is None or df.empty:
            return pd.Series()
        
        # 1. レースコード系カラムでの完全一致検索（小数点や型揺れを完全修復）
        for col in df.columns:
            if any(k in col.lower() for k in ["レースコード", "rcd", "code", "r_code", "id"]):
                clean_series = df[col].astype(str).str.strip().str.split('.').str[0].str.lstrip("0")
                target_clean = str(target_code).strip().split('.')[0].lstrip("0")
                matched = df[clean_series == target_clean]
                if not matched.empty:
                    return matched.iloc[0]

        # 2. 会場コードとレース番号の個別カラムでの一致検索
        venue_col = None
        r_col = None
        for col in df.columns:
            col_l = col.lower()
            if any(k in col_l for k in ["場", "venue", "jyo", "jcd"]):
                venue_col = col
            if any(k in col_l for k in ["r", "レース", "race_num", "round"]):
                r_col = col
                
        if venue_col and r_col:
            matched = df[
                (df[venue_col].astype(str).str.strip().str.zfill(2) == str(venue_code).zfill(2)) & 
                (df[r_col].astype(str).str.strip().str.zfill(2) == str(r_num).zfill(2))
            ]
            if not matched.empty:
                return matched.iloc[0]

        # 3. 文字列の部分一致検索
        for col in df.columns:
            if any(k in col.lower() for k in ["レース", "rcd", "code"]):
                matched = df[df[col].astype(str).str.contains(venue_s) & df[col].astype(str).str.contains(r_str)]
                if not matched.empty:
                    return matched.iloc[0]
                    
        return df.iloc[0] if len(df) > 0 else pd.Series()

    prog_row = extract_target_row(df_cards)
    sui_row = extract_target_row(df_sui)
    orig_row = extract_target_row(df_orig)

    def get_val(row, boat_num, field_names, default=0.0):
        if row is None or row.empty:
            return default
        if isinstance(field_names, str):
            field_names = [field_names]
            
        prefixes = [
            f"艇{boat_num}_", f"{boat_num}_", f"艇{boat_num}", f"{boat_num}", 
            f"boat{boat_num}_", f"b{boat_num}_", f"p{boat_num}_", f"player{boat_num}_"
        ]
        suffixes = [
            f"_{boat_num}", f"艇{boat_num}", f"{boat_num}"
        ]
        
        row_keys = {str(k).strip(): k for k in row.index}
        
        for fn in field_names:
            if fn in row_keys:
                val = row[row_keys[fn]]
                if pd.notna(val) and str(val).strip() != "":
                    return val
            for p in prefixes:
                for candidate in [f"{p}{fn}", f"{fn}{p}"]:
                    if candidate in row_keys:
                        val = row[row_keys[candidate]]
                        if pd.notna(val) and str(val).strip() != "":
                            return val
            for s in suffixes:
                candidate = f"{fn}{s}"
                if candidate in row_keys:
                    val = row[row_keys[candidate]]
                    if pd.notna(val) and str(val).strip() != "":
                        return val
            for rk_str, orig_k in row_keys.items():
                if str(boat_num) in rk_str and fn in rk_str:
                    val = row[orig_k]
                    if pd.notna(val) and str(val).strip() != "":
                        return val
        return default

    combined_rows = []
    
    for boat_num in range(1, 7):
        row_dict = {
            "レース場": float(venue_code) if str(venue_code).isdigit() else 0.0,
            "艇番": float(boat_num),
            "枠番": float(boat_num),
            
            "風速(m)": float(get_val(sui_row, boat_num, ["風速(m)", "風速", "wind_speed"], 0.0) or 0.0),
            "風向": float(get_val(sui_row, boat_num, ["風向", "wind_dir"], 0.0) or 0.0),
            "波の高さ(cm)": float(get_val(sui_row, boat_num, ["波の高さ(cm)", "波高", "wave"], 0.0) or 0.0),
            "天候": float(get_val(sui_row, boat_num, ["天候", "weather"], 0.0) or 0.0),
            "気温(℃)": float(get_val(sui_row, boat_num, ["気温(℃)", "気温", "air_temp"], 0.0) or 0.0),
            "水温(℃)": float(get_val(sui_row, boat_num, ["水温(℃)", "水温", "water_temp"], 0.0) or 0.0),
            
            "全国勝率": float(get_val(prog_row, boat_num, ["全国勝率", "勝率", "national_win_rate"], 0.0) or 0.0),
            "全国2連対率": float(get_val(prog_row, boat_num, ["全国2連対率", "2連対率", "national_2ren"], 0.0) or 0.0),
            "全国3連対率": float(get_val(prog_row, boat_num, ["全国3連対率", "3連対率", "national_3ren"], 0.0) or 0.0),
            "当地勝率": float(get_val(prog_row, boat_num, ["当地勝率", "local_win_rate"], 0.0) or 0.0),
            "当地2連対率": float(get_val(prog_row, boat_num, ["当地2連対率", "local_2ren"], 0.0) or 0.0),
            "当地3連対率": float(get_val(prog_row, boat_num, ["当地3連対率", "local_3ren"], 0.0) or 0.0),
            "モーター2連率": float(get_val(prog_row, boat_num, ["モーター2連率", "モータ2連率", "motor_2ren"], 0.0) or 0.0),
            "モーター3連率": float(get_val(prog_row, boat_num, ["モーター3連率", "モータ3連率", "motor_3ren"], 0.0) or 0.0),
            "ボート2連率": float(get_val(prog_row, boat_num, ["ボート2連率", "boat_2ren"], 0.0) or 0.0),
            "ボート3連率": float(get_val(prog_row, boat_num, ["ボート3連率", "boat_3ren"], 0.0) or 0.0),
            "全国平均ST": float(get_val(prog_row, boat_num, ["全国平均ST", "平均ST", "st", "avg_st"], 0.15) or 0.15),
            "F本数": float(get_val(prog_row, boat_num, ["F本数", "f_count"], 0.0) or 0.0),
            "L本数": float(get_val(prog_row, boat_num, ["L本数", "l_count"], 0.0) or 0.0),
            "年齢": float(get_val(prog_row, boat_num, ["年齢", "age"], 0.0) or 0.0),
            "級別": str(get_val(prog_row, boat_num, ["級別", "class"], "B1")),
            "選手名": str(get_val(prog_row, boat_num, ["選手名", "name", "player"], f"選手{boat_num}")),
            "登録番号": get_val(prog_row, boat_num, ["登録番号", "id", "reg_no"], 0),
        }
        
        v1 = get_val(orig_row, boat_num, ["値1", "ex_val1", "val1", "評価1"], 0.0)
        v2 = get_val(orig_row, boat_num, ["値2", "ex_val2", "val2", "評価2"], 0.0)
        v3 = get_val(orig_row, boat_num, ["値3", "ex_val3", "val3", "評価3"], 0.0)
        item1 = str(get_val(orig_row, boat_num, ["計測項目1", "item1"], ""))
        item2 = str(get_val(orig_row, boat_num, ["計測項目2", "item2"], ""))
        item3 = str(get_val(orig_row, boat_num, ["計測項目3", "item3"], ""))
        
        row_dict["回り足"] = float(get_val(orig_row, boat_num, ["回り足", "まわり足", "turn"], 0.0) or 0.0)
        row_dict["直線"] = float(get_val(orig_row, boat_num, ["直線", "straight"], 0.0) or 0.0)
        row_dict["一周タイム"] = float(get_val(orig_row, boat_num, ["一周タイム", "lap1"], 0.0) or 0.0)
        row_dict["半周タイム"] = float(get_val(orig_row, boat_num, ["半周タイム", "lap0.5"], 0.0) or 0.0)
        
        for item, val in zip([item1, item2, item3], [v1, v2, v3]):
            try:
                val_f = float(val)
            except:
                val_f = 0.0
            item_str = str(item)
            if "まわり足" in item_str or "回り足" in item_str:
                row_dict["回り足"] = val_f
            elif "直線" in item_str:
                row_dict["直線"] = val_f
            elif "一周" in item_str:
                row_dict["一周タイム"] = val_f
            elif "半周" in item_str:
                row_dict["半周タイム"] = val_f

        combined_rows.append(row_dict)

    df_input = pd.DataFrame(combined_rows)
    
    rank_map = {"A1": 4, "A2": 3, "B1": 2, "B2": 1}
    if "級別" in df_input.columns:
        df_input["級別"] = df_input["級別"].map(rank_map).fillna(2)

    if player_course_stats is not None and "選手名" in df_input.columns and "枠番" in df_input.columns:
        df_input = pd.merge(df_input, player_course_stats, on=["選手名", "枠番"], how="left")
    if venue_wind_kimarite is not None and "レース場" in df_input.columns and "風向" in df_input.columns:
        df_input = pd.merge(df_input, venue_wind_kimarite, on=["レース場", "風向"], how="left")
    if player_fav_kimarite is not None and "選手名" in df_input.columns:
        df_input = pd.merge(df_input, player_fav_kimarite, on=["選手名"], how="left")

    for col in df_input.columns:
        if not df_input[col].dtype.name.startswith("cat") and col != "選手名":
            df_input[col] = pd.to_numeric(df_input[col], errors="coerce")

    X_input = df_input.reindex(columns=expected_features, fill_value=0.0)

    prob_matrix = {}
    for rank_idx, rank_name in enumerate(["rank_1", "rank_2", "rank_3"]):
        if rank_name in models:
            pred_val = models[rank_name].predict(X_input)
            prob_matrix[rank_idx + 1] = np.ravel(pred_val)

    boat_data = []
    for i in range(len(df_input)):
        boat_num = i + 1
        row_d = df_input.iloc[i]
        name = str(row_d.get("選手名", f"選手{boat_num}"))
        
        arr_1 = prob_matrix.get(1, np.zeros(6))
        arr_2 = prob_matrix.get(2, np.zeros(6))
        arr_3 = prob_matrix.get(3, np.zeros(6))
        
        p1 = float(arr_1[i]) * 100 if len(arr_1) > i else 0.0
        p2 = float(arr_2[i]) * 100 if len(arr_2) > i else 0.0
        p3 = float(arr_3[i]) * 100 if len(arr_3) > i else 0.0
        
        boat_data.append({"boat": boat_num, "name": name, "p1": p1, "p2": p2, "p3": p3})

    if not boat_data:
        return summary_text + " ⚠️ エラー: 艇データが取得できませんでした。"

    top_1st = max(boat_data, key=lambda x: x['p1'])
    top_2nd = max(boat_data, key=lambda x: x['p2'])

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

