                # --- 【厳選・中穴特化型】オッズ30〜50倍かつ、予測確率や条件が揃ったレースのみに絞る ---
                target_odds_combos = {}
                for k, p in comb_probs.items():
                    if k in raw_odds:
                        odds_val = raw_odds[k]
                        # 30〜50倍かつ、ある程度確証のある確率（例: 0.015以上など）に絞る
                        if 30.0 <= odds_val <= 50.0 and p >= 0.015:
                            target_odds_combos[k] = p * odds_val

                # 旨味と確証が揃った買い目がなければ、このレースは勇気を持ってパス（見送り）
                if len(target_odds_combos) < 2:
                    continue

