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
    "13": "ana", "14": "nar", "15": "mar", "16": "koj", "17": "miy", "18": "tok",
    "19": "shm", "20": "wkm", "21": "ash", "22": "fuk", "23": "ktu", "24": "oom"
}

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

CSV_CACHE = {}

def fetch_github_csv(file_path, use_cache=False):
    """
    GitHubからCSVを取得する。
    use_cache=True の場合は静的マスタをキャッシュ。
    use_cache=False の場合はタイムスタンプを付与してリアルタイムデータを確実に取得。
    """
    if use_cache and file_path in CSV_CACHE:
        return CSV_CACHE[file_path]
    
    timestamp = int(datetime.now().timestamp())
    uri = f"{GITHUB_RAW_BASE}{file_path}?t={timestamp}"
    try:
        res = requests.get(uri, timeout=10)
        if res.status_code == 200:
            df = pd.read_csv(io.StringIO(res.text), encoding="utf-8-sig")
            df.columns = df.columns.str.strip()
            if use_cache:
                CSV_CACHE[file_path] = df
            return df
    except Exception as e:
        print(f"CSV Fetch Error ({file_path}): {e}")
    return None

def fetch_github_csv_with_fallback(primary_path, fallback_path, use_cache=False):
    df = fetch_github_csv(primary_path, use_cache=use_cache)
    if df is None and fallback_path:
        df = fetch_github_csv(fallback_path, use_cache=use_cache)
    return df

# --- モデルおよびデータの読み込み ---
MODEL_FILENAME = "boatrace_lgb_model.pkl"
models = {}
player_fav_kimarite = None
kimarite_prob_dict = {}
expected_features = []
feature_medians = {}

try:
    if os.path.exists(MODEL_FILENAME):
        loaded_package = joblib.load(MODEL_FILENAME)
        if isinstance(loaded_package, dict):
            if "model_1st" in loaded_package:
                models["rank_1"] = loaded_package.get("model_1st")
                models["rank_2"] = loaded_package.get("model_2nd")
                models["rank_3"] = loaded_package.get("model_3rd")
            elif "models" in loaded_package:
                models = loaded_package.get("models")
            elif "rank_1" in loaded_package:
                models = loaded_package

            player_fav_kimarite = loaded_package.get("player_fav_kimarite")
            loaded_pair_table = loaded_package.get("pair_table")
            if loaded_pair_table and isinstance(loaded_pair_table, dict):
                kimarite_prob_dict = loaded_pair_table

            if "feature_names" in loaded_package:
                expected_features = loaded_package["feature_names"]
            elif "features" in loaded_package:
                expected_features = loaded_package["features"]
            elif "rank_1" in models and hasattr(models["rank_1"], "feature_name"):
                expected_features = models["rank_1"].feature_name()

            if "feature_medians" in loaded_package:
                feature_medians = loaded_package.get("feature_medians", {})

            print("パッケージ形式でモデルとデータを正常に読み込みました。")
        else:
            models = loaded_package
            print("モデル単体として読み込みました。")
            if "rank_1" in models and hasattr(models["rank_1"], "feature_name"):
                expected_features = models["rank_1"].feature_name()
    else:
        print(f"警告: モデルファイル ('{MODEL_FILENAME}') が見つかりません。")
except Exception as e:
    models = {}
    print(f"モデルの読み込みに失敗しました: {e}")

def load_kimarite_table_from_github():
    global kimarite_prob_dict
    if kimarite_prob_dict:
        return
    df_pair = fetch_github_csv("data/estimate/kimarite/tables/pair_table.csv", use_cache=True)
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
    header_text = f"🎯 **{venue}** {r_num}R 予想結果 ({day_str})\n"

    if not models or "rank_1" not in models or models["rank_1"] is None:
        return header_text + "⚠️ エラー: 予測モデルが読み込まれていません。"

    day_part = day_str.split("-")[2] if "-" in day_str else day_str
    month_str = str(int(month)).zfill(2)
    day_str_zf = str(int(day_part)).zfill(2)
    month_raw = str(int(month))
    day_raw = str(int(day_part))

    venue_s = str(venue_code).zfill(2)
    month_int = int(month)

    race_card_p1 = f"data/programs/race_cards/{year}/{month_str}/{day_str_zf}.csv"
    race_card_p2 = f"data/programs/race_cards/{year}/{month_raw}/{day_raw}.csv"

    sui_p1 = f"data/previews/sui/{year}/{month_str}/{day_str_zf}.csv"
    sui_p2 = f"data/previews/sui/{year}/{month_raw}/{day_raw}.csv"

    orig_p1 = f"data/previews/original_exhibition/{year}/{month_str}/{day_str_zf}.csv"
    orig_p2 = f"data/previews/original_exhibition/{year}/{month_raw}/{day_raw}.csv"

    stt_p1 = f"data/previews/stt/{year}/{month_str}/{day_str_zf}.csv"
    stt_p2 = f"data/previews/stt/{year}/{month_raw}/{day_raw}.csv"

    prev_code = VENUE_PREVIEW_CODE_MAP.get(venue_s, "")
    venue_preview_p1 = f"data/previews/{prev_code}/{year}/{month_str}/{day_str_zf}.csv" if prev_code else None
    venue_preview_p2 = f"data/previews/{prev_code}/{year}/{month_raw}/{day_raw}.csv" if prev_code else None

    # 展示・直前データはリアルタイム取得
    df_cards = fetch_github_csv_with_fallback(race_card_p1, race_card_p2, use_cache=False)
    df_sui = fetch_github_csv_with_fallback(sui_p1, sui_p2, use_cache=False)
    df_orig = fetch_github_csv_with_fallback(orig_p1, orig_p2, use_cache=False)
    df_stt = fetch_github_csv_with_fallback(stt_p1, stt_p2, use_cache=False)
    df_venue_preview = fetch_github_csv_with_fallback(venue_preview_p1, venue_preview_p2, use_cache=False)

    if df_orig is not None:
        rename_dict = {f"艇{i}_値{j}": f"艇{i}_オリジナル" + ["一周タイム", "まわり足タイム", "直線タイム"][j-1] for i in range(1, 7) for j in range(1, 4)}
        df_orig = df_orig.rename(columns=rename_dict)

    df_course_win = fetch_github_csv("data/estimate/stadium/course_win_rate.csv", use_cache=True)
    df_season_win = fetch_github_csv("data/estimate/stadium/win_rate.csv", use_cache=True)

    if df_cards is None:
        return header_text + "⚠️ エラー: 出走表データが取得できませんでした。"

    r_str = str(r_num).zfill(2)
    target_race_code = f"{year}{month_str}{day_str_zf}{venue_s}{r_str}"

    def get_matched_row(df, code):
        if df is None: return None
        for col in df.columns:
            if "レースコード" in col or "code" in col.lower():
                col_vals = df[col].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
                matched = df[col_vals == str(code)]
                if len(matched) > 0:
                    return matched.iloc[0].to_dict()
        return None

    card_row = get_matched_row(df_cards, target_race_code)
    if not card_row:
        return header_text + f"⚠️ エラー: レースコード '{target_race_code}' のデータが見つかりません。"

    combined_row = {}
    combined_row.update(card_row)
    combined_row.update(get_matched_row(df_sui, target_race_code) or {})
    combined_row.update(get_matched_row(df_orig, target_race_code) or {})
    combined_row.update(get_matched_row(df_stt, target_race_code) or {})
    combined_row.update(get_matched_row(df_venue_preview, target_race_code) or {})

    if df_course_win is not None:
        for _, row in df_course_win.iterrows():
            if str(row.get("場コード", "")).strip().zfill(2) == venue_s and str(row.get("レース回", "")).strip() == str(int(r_num)):
                for k, v in row.items():
                    if k not in ["場コード", "レース回"]: combined_row[f"est_course_{k}"] = v
                break

    if df_season_win is not None:
        season_name = get_season(month_int)
        for _, row in df_season_win.iterrows():
            if str(row.get("場コード", "")).strip().zfill(2) == venue_s and str(row.get("季節", "")).strip() == season_name:
                for k, v in row.items():
                    if k not in ["場コード", "季節"]: combined_row[f"est_season_{k}"] = v
                break

    expanded_row = dict(combined_row)
    for k, v in list(combined_row.items()):
        for b in range(1, 7):
            sb = str(b)
            if k.startswith(f"艇{sb}_"):
                expanded_row[f"{sb}号艇_{k[2:]}"] = v
                expanded_row[f"{k[2:]}_{sb}"] = v
            elif k.startswith(f"{sb}号艇_"):
                expanded_row[f"艇{sb}_{k[3:]}"] = v
                expanded_row[f"{k[3:]}_{sb}"] = v
            elif k.endswith(f"_{sb}"):
                expanded_row[f"艇{sb}_{k[:-2]}"] = v
                expanded_row[f"{sb}号艇_{k[:-2]}"] = v

    df_pred = pd.DataFrame([expanded_row])

    if player_fav_kimarite:
        for i in range(1, 7):
            p_col_candidates = [f"艇{i}_選手名", f"{i}号艇_選手名", f"選手名_{i}", f"艇{i}_氏名", f"{i}号艇_氏名", f"氏名_{i}", f"艇{i}_選手", f"{i}号艇_選手"]
            p_col = next((c for c in p_col_candidates if c in df_pred.columns), None)
            if p_col:
                dummy_k_keys = list(next(iter(player_fav_kimarite.values())).keys()) if player_fav_kimarite else []
                p_val = str(df_pred.iloc[0].get(p_col, "")).strip()
                for k_name in dummy_k_keys:
                    df_pred[f"艇{i}_kimarite_{k_name}"] = player_fav_kimarite.get(p_val, {}).get(k_name, 0.0)

    X_input = df_pred.reindex(columns=expected_features)

    for col in expected_features:
        if col in ["レース場", "風向", "天候"] and col in X_input.columns:
            X_input[col] = X_input[col].astype('category')

    for col in X_input.columns:
        if col not in ["レース場", "風向", "天候"]:
            X_input[col] = pd.to_numeric(X_input[col], errors='coerce')
            X_input[col] = X_input[col].fillna(feature_medians.get(col, 0.0) if isinstance(feature_medians, dict) else 0.0)

    prob_matrix = {}
    for rank_idx, rank_name in enumerate(["rank_1", "rank_2", "rank_3"], 1):
        if rank_name in models and models[rank_name] is not None:
            preds = models[rank_name].predict(X_input)
            if len(preds) > 0:
                prob_matrix[rank_idx] = np.array(preds[0])

    boat_names = []
    for i in range(6):
        b_num = i + 1
        name = f"選手{b_num}"
        for c in [f"艇{b_num}_選手名", f"{b_num}号艇_選手名", f"選手名_{b_num}", f"艇{b_num}_氏名", f"{b_num}号艇_氏名", f"氏名_{b_num}", f"艇{b_num}_選手", f"{b_num}号艇_選手"]:
            if c in df_pred.columns and pd.notna(df_pred.iloc[0][c]):
                val = str(df_pred.iloc[0][c]).strip()
                if val and val != "nan":
                    name = val
                    break
        boat_names.append(name)

    summary_text = f"🎯 **{venue}** {r_num}R 予想結果 ({day_str})\n\n"
    summary_text += "--- 【3連単 予想買い目（上位5点）】 ---\n"

    if 1 in prob_matrix and 2 in prob_matrix and 3 in prob_matrix:
        m1, m2, m3 = prob_matrix[1], prob_matrix[2], prob_matrix[3]
        entry_courses = {i+1: i+1 for i in range(6)}
        default_kimarite_map = {1: "逃げ", 2: "差し", 3: "まくり", 4: "まくり", 5: "まくり差し", 6: "まくり差し"}

        trifecta_scores = []
        for c1_idx, c2_idx, c3_idx in itertools.permutations(range(6), 3):
            b1, b2, b3 = c1_idx + 1, c2_idx + 1, c3_idx + 1
            p1, p2, p3 = float(m1[c1_idx]), float(m2[c2_idx]), float(m3[c3_idx])

            ai_base_score = (p1 ** 1.8) * (p2 ** 1.3) * (p3 ** 1.0)
            c1_course, c2_course, c3_course = entry_courses[b1], entry_courses[b2], entry_courses[b3]
            primary_kimarite = default_kimarite_map.get(b1, "差し")
            k_key = f"{primary_kimarite}_{c1_course}"
            pair_prob = kimarite_prob_dict.get((k_key, c2_course, c3_course), kimarite_prob_dict.get((primary_kimarite, c2_course, c3_course), 0.001))

            final_score = ai_base_score * (max(pair_prob, 0.001) ** 0.3)
            trifecta_scores.append(((b1, b2, b3), final_score))

        trifecta_scores.sort(key=lambda x: x[1], reverse=True)

        for rank, (combo, score) in enumerate(trifecta_scores[:5], 1):
            summary_text += f"第{rank}位: **{combo[0]} - {combo[1]} - {combo[2]}** (スコア: {score:.4f})\n"

    summary_text += "\n--- 【各艇の予測確率】 ---\n"
    arr_1 = prob_matrix.get(1, np.zeros(6))
    arr_2 = prob_matrix.get(2, np.zeros(6))
    arr_3 = prob_matrix.get(3, np.zeros(6))

    for i in range(6):
        p1 = float(arr_1[i]) * 100 if len(arr_1) > i else 0.0
        p2 = float(arr_2[i]) * 100 if len(arr_2) > i else 0.0
        p3 = float(arr_3[i]) * 100 if len(arr_3) > i else 0.0

        p_2ren = min(p1 + p2, 100.0)
        p_3ren = min(p1 + p2 + p3, 100.0)

        summary_text += f"・{i+1}号艇 {boat_names[i]}: 1着率 {p1:.1f}% | 2連率 {p_2ren:.1f}% | 3連率 {p_3ren:.1f}%\n"

    return summary_text

# --- Discord UI部分 ---

class InteractiveRaceControlView(discord.ui.View):
    def __init__(self, current_venue: str):
        super().__init__(timeout=None)
        self.current_venue = current_venue

        options = [discord.SelectOption(label=v, description=f"{v} のレース予想を表示") for v in VENUES]
        venue_select = discord.ui.Select(
            placeholder=f"🏟️ 現在: {current_venue} (会場変更はこちら)",
            min_values=1, max_values=1,
            options=options,
            custom_id=f"p_venue_select_{current_venue}"
        )
        venue_select.callback = self.venue_callback
        self.add_item(venue_select)

        all_btn = discord.ui.Button(
            label="⭐ 全12R一括予想",
            style=discord.ButtonStyle.success,
            custom_id=f"p_all_btn_{current_venue}"
        )
        all_btn.callback = self.all_callback
        self.add_item(all_btn)

        for r in range(1, 13):
            r_btn = discord.ui.Button(
                label=f"{r}R",
                style=discord.ButtonStyle.primary,
                custom_id=f"p_r_btn_{current_venue}_{r}"
            )
            r_btn.callback = self.make_race_callback(r)
            self.add_item(r_btn)

    def make_race_callback(self, r_num: int):
        async def callback(interaction: discord.Interaction):
            await self.race_callback(interaction, r_num)
        return callback

    async def venue_callback(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)
        selected_venue = interaction.data["values"][0]
        await interaction.followup.send(
            content=f"🏟️ **{self.current_venue}** から **{selected_venue}** に切り替えました。",
            view=InteractiveRaceControlView(selected_venue),
            ephemeral=True
        )

    async def all_callback(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)
        venue = self.current_venue
        venue_code = VENUE_MAPPING.get(venue, "01")
        target_date = datetime.now(JST)
        year, month, day_str = target_date.strftime("%Y"), target_date.strftime("%m"), target_date.strftime("%Y-%m-%d")

        await interaction.followup.send(content=f"🤖 **{venue}** 全12レースの解析を開始します...", ephemeral=True)
        for r_num in range(1, 13):
            res_text = calculate_single_race_analysis(venue, venue_code, year, month, day_str, r_num)
            await interaction.followup.send(content=res_text, ephemeral=True)
            await asyncio.sleep(0.2)

    async def race_callback(self, interaction: discord.Interaction, r_num: int):
        await interaction.response.defer(ephemeral=True)
        try:
            venue = self.current_venue
            venue_code = VENUE_MAPPING.get(venue, "01")
            target_date = datetime.now(JST)
            year, month, day_str = target_date.strftime("%Y"), target_date.strftime("%m"), target_date.strftime("%Y-%m-%d")

            result_text = calculate_single_race_analysis(venue, venue_code, year, month, day_str, r_num)
            await interaction.followup.send(content=result_text, ephemeral=True)
        except Exception as e:
            tb = traceback.format_exc()
            await interaction.followup.send(content=f"⚠️ エラーが発生しました:\n```python\n{tb}\n```", ephemeral=True)

class VenueSelect(discord.ui.Select):
    def __init__(self):
        options = [discord.SelectOption(label=v, description=f"{v} のレース予想を表示") for v in VENUES]
        super().__init__(
            placeholder="最初にする会場を選択してください...",
            min_values=1,
            max_values=1,
            options=options,
            custom_id="persistent_venue_select_init"
        )

    async def callback(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)
        venue = self.values[0]
        await interaction.followup.send(
            content=f"🏟️ **{venue}** が選択されました。このメニューからいつでも操作できます。",
            view=InteractiveRaceControlView(venue),
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
        for v in VENUES:
            bot.add_view(InteractiveRaceControlView(v))
    except Exception as e:
        print(f"Error adding view: {e}")

@bot.command(name="setup")
async def setup(ctx):
    await ctx.message.delete()
    await ctx.send(
        content="🤖 **【AIレース分析・予想メニュー】**\n👇 下のメニューから会場を選択してください：",
        view=VenueSelectView()
    )

@bot.command(name="status")
async def check_status(ctx):
    test_files = [
        "data/estimate/kimarite/tables/pair_table.csv",
        "data/estimate/stadium/course_win_rate.csv",
        "data/estimate/stadium/win_rate.csv"
    ]
    file_results = [f"✅ {f} (取得成功: {len(fetch_github_csv(f, use_cache=True))}行)" if fetch_github_csv(f, use_cache=True) is not None else f"❌ {f} (取得失敗)" for f in test_files]

    msg = (
        f"📊 **【データ取り込み状況チェック】**\n\n"
        f"***1. モデル読み込み状態**: {'成功' if models and 'rank_1' in models else '失敗'}\n"
        f"***2. モデルの特徴量数**: {len(expected_features)} 個\n"
        f"***3. 決まり手テーブル件数**: {len(kimarite_prob_dict)} 件\n"
        f"***4. GitHubファイル取得テスト**:\n" + "\n".join(file_results)
    )
    await ctx.send(msg)

if __name__ == "__main__":
    keep_alive()
    token = os.environ.get("DISCORD_TOKEN") or os.environ.get("DISCORD_BOT_TOKEN")
    if token:
        bot.run(token)
    else:
        print("エラー: DISCORD_TOKEN 環境変数が設定されていません。")
  
