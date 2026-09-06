import numpy as np
import pandas as pd

class V12LongshotSkewPredictor:
    """
    荒れ度フィルター・締切直前オッズ歪み・動的資金管理・変動制（最大10点）に加え、
    最大ドローダウン（MDD）の計測機能を持つ回収率800%ターゲットの穴特化プレディクター
    """
    def __init__(self, predictor_id="v12_longshot_skew", max_points=10, target_roi_multiplier=8.0):
        self.predictor_id = predictor_id
        self.max_points = max_points                    # 最大10点
        self.target_roi_multiplier = target_roi_multiplier  # 目標回収倍率（800% = 8.0倍）
        self.absolute_min_odds = 40.0                   # 穴とみなす最低オッズ

    def evaluate_race(self, model_probs, odds_dict, volatility_score=1.5):
        """
        実戦・リアルタイム予測用の評価関数
        """
        if volatility_score < 1.2:
            return {"status": "skipped", "reason": "Low volatility / firm race expected"}

        candidates = []
        for combination, p_model in model_probs.items():
            odds = odds_dict.get(combination, 0.0)
            
            if odds < self.absolute_min_odds:
                continue
                
            expected_value = p_model * odds
            if expected_value >= 1.0:
                candidates.append({
                    "combination": combination,
                    "model_prob": p_model,
                    "odds": odds,
                    "expected_value": expected_value
                })

        if not candidates:
            return {"status": "skipped", "reason": "No valid longshot candidates"}

        candidates = sorted(candidates, key=lambda x: x["expected_value"], reverse=True)

        raw_picks = candidates[:self.max_points]
        num_points = len(raw_picks)
        required_odds_threshold = num_points * self.target_roi_multiplier

        final_picks = []
        for cand in raw_picks:
            if cand["odds"] >= max(self.absolute_min_odds, required_odds_threshold):
                final_picks.append(cand)

        if not final_picks:
            return {"status": "skipped", "reason": "Failed ROI multiplier threshold"}

        total_ev = sum(p["expected_value"] for p in final_picks)
        for cand in final_picks:
            cand["bet_weight"] = cand["expected_value"] / total_ev if total_ev > 0 else (1.0 / len(final_picks))

        return {
            "status": "active",
            "num_points": len(final_picks),
            "picks": final_picks
        }

    def backtest_simulation(self, historical_races_data, initial_bankroll=1000000):
        """
        最大ドローダウン（MDD）および累積収支の推移を計算するバックテスト機能
        """
        current_bankroll = initial_bankroll
        peak_bankroll = initial_bankroll
        max_drawdown = 0.0
        max_drawdown_rate = 0.0

        total_investment = 0
        total_payout = 0
        results_log = []

        for race in historical_races_data:
            evaluation = self.evaluate_race(
                race["probs"], 
                race["odds"], 
                race.get("volatility", 1.5)
            )
            
            if evaluation["status"] == "skipped":
                continue

            picks = evaluation["picks"]
            race_budget = 10000  # 1レースあたりの基準投資額
            hit = False
            race_investment = 0
            race_payout = 0

            for pick in picks:
                stake = race_budget * pick["bet_weight"]
                race_investment += stake
                total_investment += stake
                
                if pick["combination"] == race["actual_result"]:
                    hit = True
                    race_payout += stake * pick["odds"]

            total_payout += race_payout
            
            # バンクロール（資金）の更新
            current_bankroll = current_bankroll - race_investment + race_payout

            # ピーク資金の更新とドローダウンの計算
            if current_bankroll > peak_bankroll:
                peak_bankroll = current_bankroll
            
            drawdown = peak_bankroll - current_bankroll
            drawdown_rate = (drawdown / peak_bankroll) if peak_bankroll > 0 else 0.0

            if drawdown > max_drawdown:
                max_drawdown = drawdown
            if drawdown_rate > max_drawdown_rate:
                max_drawdown_rate = drawdown_rate

            results_log.append({
                "race_id": race.get("id"),
                "hit": hit,
                "investment": race_investment,
                "payout": race_payout,
                "bankroll": current_bankroll,
                "drawdown": drawdown
            })

        overall_roi = (total_payout / total_investment * 100.0) if total_investment > 0 else 0.0
        
        return {
            "initial_bankroll": initial_bankroll,
            "final_bankroll": current_bankroll,
            "total_investment": total_investment,
            "total_payout": total_payout,
            "roi": overall_roi,
            "max_drawdown": max_drawdown,
            "max_drawdown_rate": max_drawdown_rate * 100.0, # パーセンテージ表示
            "logs": results_log
        }

