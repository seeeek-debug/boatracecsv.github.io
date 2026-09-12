import os
import io
import threading
from flask import Flask
import discord
from discord.ext import commands
import pandas as pd
import requests
import joblib

# --- Render用Webサーバー（ポートを最速で開く） ---
app = Flask(__name__)

@app.route("/")
def home():
    return "I am alive"

def run_web():
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, use_reloader=False)

threading.Thread(target=run_web, daemon=True).start()

# --- 設定・初期化 ---
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/seeeek-debug/boatracecsv.github.io/main/"
MODEL_FILENAME = "boatrace_lgb_model.pkl"

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

# データのロード状態を保持する辞書
status_log = {
    "model": "未ロード",
    "features_count": 0,
    "pair_table": "未ロード"
}

models = None
expected_features = []

# モデルの読み込みテスト
try:
    loaded = joblib.load(MODEL_FILENAME)
    if isinstance(loaded, dict):
        models = loaded.get("models")
        if models and "rank_1" in models:
            expected_features = models["rank_1"].feature_name()
            status_log["model"] = "成功 (dict形式)"
            status_log["features_count"] = len(expected_features)
    else:
        models = loaded
        status_log["model"] = "成功 (単体形式)"
except Exception as e:
    status_log["model"] = f"失敗: {e}"

@bot.event
async def on_ready():
    print(f"Logged in as {bot.user.name}")

@bot.command(name="status")
async def check_status(ctx):
    """何が取り込めて、何が取り込めていないかを確認するコマンド"""
    
    # GitHub上の主要CSVが取得できるかテスト
    test_files = [
        "pair_table.csv",
        "data/estimate/stadium/course_win_rate.csv",
        "data/estimate/stadium/win_rate.csv"
    ]
    
    file_results = []
    for fpath in test_files:
        try:
            res = requests.get(f"{GITHUB_RAW_BASE}{fpath}", timeout=5)
            if res.status_code == 200:
                file_results.append(f"✅ {fpath} (取得成功)")
            else:
                file_results.append(f"❌ {fpath} (ステータスコード: {res.status_code})")
        except Exception as e:
            file_results.append(f"❌ {fpath} (エラー: {e})")

    files_text = "\n".join(file_results)

    msg = (
        f"📊 **【データ取り込み状況チェック】**\n\n"
        f"**1. モデル読み込み状態**: {status_log['model']}\n"
        f"**2. モデルの特徴量数**: {status_log['features_count']} 個\n\n"
        f"**3. GitHubファイル取得テスト**:\n{files_text}"
    )

    await ctx.send(msg)

if __name__ == "__main__":
    token = os.environ.get("DISCORD_TOKEN")
    bot.run(token)

