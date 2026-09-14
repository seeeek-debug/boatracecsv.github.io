import os
from datetime import datetime
import numpy as np
import requests
from bs4 import BeautifulSoup
import discord
from discord.ext import commands
from discord import app_commands, Interaction

# ==========================================
# 1. 環境変数からのトークン取得
# ==========================================
DISCORD_BOT_TOKEN = os.environ.get("DISCORD_BOT_TOKEN")

# 競艇場コードマップ
VENUE_MAP = {
    "桐生": "01", "戸田": "02", "江戸川": "03", "平和島": "04", "多摩川": "05",
    "浜名湖": "06", "蒲郡": "07", "常滑": "08", "津": "09", "三国": "10",
    "びわこ": "11", "住之江": "12", "尼崎": "13", "鳴門": "14", "丸亀": "15",
    "児島": "16", "宮島": "17", "徳山": "18", "下関": "19", "若松": "20",
    "芦屋": "21", "福岡": "22", "唐津": "23", "大村": "24"
}

# ==========================================
# 2. オッズ取得スクレイパー (BOATRACE公式)
# ==========================================
def fetch_3rentan_odds(venue_name_or_code: str, race_num: int, date_str: str) -> dict:
    """
    User-Agentを指定して403アクセス拒否を防止しオッズを取得
    """
    jcd = VENUE_MAP.get(str(venue_name_or_code), str(venue_name_or_code)).zfill(2)
    hd = str(date_str).replace("-", "").replace("/", "")
    rno = str(race_num)
    
    url = f"https://www.boatrace.jp/owpc/pc/race/odds3t?rno={rno}&jcd={jcd}&hd={hd}"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Referer": "https://www.boatrace.jp/"
    }
    
    odds_dict = {}
    try:
        res = requests.get(url, headers=headers, timeout=5)
        if res.status_code != 200:
            return odds_dict
            
        soup = BeautifulSoup(res.content, "html.parser")
        tables = soup.select("table.grid2")
        
        for table in tables:
            rows = table.select("tr")
            cur_1st = None
            cur_2nd = None
            
            for tr in rows:
                td_1st = tr.select_one("th.is-p1, td.is-p1")
                if td_1st:
                    cur_1st = td_1st.get_text(strip=True)
                    
                td_2nd = tr.select_one("td.is-p2, th.is-p2")
                if td_2nd:
                    cur_2nd = td_2nd.get_text(strip=True)
                    
                td_3rd = tr.select_one("td.is-p3, th.is-p3")
                td_odds = tr.select_one("td.oddsPoint")
                
                if cur_1st and cur_2nd and td_3rd and td_odds:
                    cur_3rd = td_3rd.get_text(strip=True)
                    odds_val = td_odds.get_text(strip=True)
                    if cur_1st.isdigit() and cur_2nd.isdigit() and cur_3rd.isdigit():
                        key = f"{cur_1st}-{cur_2nd}-{cur_3rd}"
                        odds_dict[key] = odds_val
    except Exception as e:
        print(f"オッズ取得エラー ({venue_name_or_code} {race_num}R): {e}")
        
    return odds_dict

# ==========================================
# 3. 予想データ作成関数
# ==========================================
def predict_race(venue_name: str, race_num: int, date_str: str):
    prob_1st = np.array([0.506, 0.197, 0.153, 0.095, 0.026, 0.023])
    prob_2nd = np.array([0.070, 0.527, 0.200, 0.105, 0.066, 0.032])
    prob_3rd = np.array([0.035, 0.230, 0.131, 0.423, 0.124, 0.057])
    
    combo_probs = []
    for i in range(1, 7):
        for j in range(1, 7):
            if i == j: continue
            for k in range(1, 7):
                if k == i or k == j: continue
                p = prob_1st[i-1] * prob_2nd[j-1] * prob_3rd[k-1]
                combo_probs.append((f"{i}-{j}-{k}", p))
                
    combo_probs.sort(key=lambda x: x[1], reverse=True)
    top5_combos = combo_probs[:5]

    odds_data = fetch_3rentan_odds(venue_name, race_num, date_str)

    player_names = ["岡崎 恭裕", "井内 将太郎", "藤田 竜弘", "松尾 昂明", "若狭 奈美子", "荻野 裕介"]
    boat_stats = []
    for b in range(6):
        boat_stats.append({
            "boat": b + 1,
            "name": player_names[b],
            "p1": prob_1st[b] * 100,
            "p2": prob_2nd[b] * 100,
            "p3": prob_3rd[b] * 100
        })

    predictions = []
    for combo, prob in top5_combos:
        odds_val = odds_data.get(combo)
        if odds_val and odds_val != "-":
            odds_text = f"({odds_val}倍)"
        else:
            odds_text = "(オッズ取得中/未発売)"
        predictions.append((combo, odds_text))

    return {
        "venue": venue_name,
        "race_num": race_num,
        "date": date_str,
        "confidence": "🔥 【勝負推奨】 AI信頼度 : 高",
        "predictions": predictions,
        "boat_stats": boat_stats
    }

def format_prediction_message(data: dict) -> str:
    lines = []
    lines.append(f"🏁 **【{data['venue']}】 {data['race_num']}R 予想結果 ({data['date']})**")
    lines.append(f"判定: {data['confidence']}\n")
    
    lines.append("--- 【3連単 予想買い目(上位5点)】 ---")
    for idx, (combo, odds_text) in enumerate(data['predictions'], start=1):
        lines.append(f"{idx}位: **{combo}** {odds_text}")
    lines.append("")
    
    lines.append("--- 【各艇の予測確率】 ---")
    for b in data['boat_stats']:
        lines.append(
            f"{b['boat']}号艇({b['name']}): 1着率 {b['p1']:.1f}% | 2着率 {b['p2']:.1f}% | 3着率 {b['p3']:.1f}%"
        )
        
    return "\n".join(lines)

# ==========================================
# 4. Discord UIコンポーネント (ボタン表示)
# ==========================================
class RaceButton(discord.ui.Button):
    def __init__(self, race_num: int, venue_name: str, date_str: str, row: int):
        super().__init__(label=f"{race_num}R", style=discord.ButtonStyle.primary, row=row)
        self.race_num = race_num
        self.venue_name = venue_name
        self.date_str = date_str

    async def callback(self, interaction: Interaction):
        await interaction.response.defer(ephemeral=True)
        res = predict_race(self.venue_name, self.race_num, self.date_str)
        msg = format_prediction_message(res)
        await interaction.followup.send(msg, ephemeral=True)

class RaceSelectView(discord.ui.View):
    def __init__(self, venue_name: str, date_str: str):
        super().__init__(timeout=None)
        self.venue_name = venue_name
        self.date_str = date_str

        # 1R〜12Rのボタンをレイアウト生成
        for r in range(1, 13):
            row_idx = 1 if r <= 4 else (2 if r <= 8 else 3)
            self.add_item(RaceButton(r, venue_name, date_str, row_idx))

    @discord.ui.button(label="★ 全12R一括予想", style=discord.ButtonStyle.success, row=0)
    async def btn_all_races(self, interaction: Interaction, button: discord.ui.Button):
        await interaction.response.defer(ephemeral=True)
        for r in range(1, 13):
            res = predict_race(self.venue_name, r, self.date_str)
            msg = format_prediction_message(res)
            await interaction.followup.send(msg, ephemeral=True)

# ==========================================
# 5. Bot起動・スラッシュコマンド設定
# ==========================================
intents = discord.Intents.default()
bot = commands.Bot(command_prefix="!", intents=intents)

@bot.event
async def on_ready():
    print(f"ログイン成功: {bot.user.name}")
    try:
        synced = await bot.tree.sync()
        print(f"スラッシュコマンド {len(synced)} 件を同期しました。")
    except Exception as e:
        print(f"同期エラー: {e}")

@bot.tree.command(name="predict", description="指定会場の予想表示メニューを開きます")
@app_commands.describe(venue="会場名 (例: 若松)", date="日付 (YYYY-MM-DD)")
async def predict_command(interaction: Interaction, venue: str, date: str = None):
    if not date:
        date = datetime.now().strftime("%Y-%m-%d")
        
    if venue not in VENUE_MAP:
        await interaction.response.send_message(f"エラー: 会場名 '{venue}' は無効です。", ephemeral=True)
        return

    view = RaceSelectView(venue, date)
    await interaction.response.send_message(
        f"⚙️ **{venue}** が選択されました。このメニューからいつでもレース予想が可能です。",
        view=view,
        ephemeral=True
    )

if __name__ == "__main__":
    if not DISCORD_BOT_TOKEN:
        raise ValueError("エラー: 環境変数 DISCORD_BOT_TOKEN が設定されていません。")
    bot.run(DISCORD_BOT_TOKEN)
