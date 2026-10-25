# -*- coding: utf-8 -*-
"""promt_3: short_stochastic_search на 9 подвыборках пользователей.

3 способа отбора x 3 подвыборки.  Для каждой подвыборки выполняется
short_stochastic_search (адаптивный шаг 80->5), затем найденные признаки
и параметры обучаются на полном датасете.  Результат -- subset_models.csv.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd

from common import (FEATURES, FEATURE_GROUP, GPARAM, GROUPS, MEAN_W, MODELS, T0,
                    Evaluator, FeatureBuilder, build_candidates, load_tx,
                    mean_params, resolve_windows, save_csv, split_dataset)
from experiments import (exp2_filter, opt_from_params, params_from_opt,
                         stochastic_adaptive)


# ------------------------------------------------------- отбор пользователей
def user_feature_table(tx):
    """Признаки группы 1 для ВСЕХ пользователей при среднем окне W(t, n)."""
    sub = tx[(tx["ts"] >= T0 - pd.Timedelta(days=MEAN_W)) & (tx["ts"] < T0)]
    g = sub.groupby("visitor")
    cnt = g.size().astype(np.float64)
    days = sub[["visitor", "date"]].drop_duplicates().sort_values(["visitor", "date"])
    gap = days.groupby("visitor")["date"].diff()
    avg_gap = gap.groupby(days["visitor"]).mean()
    tab = pd.DataFrame({"user_tx_cnt": cnt,
                        "user_avg_active_day_gap": avg_gap,
                        "user_avg_tx": cnt / (MEAN_W / 7.0)}).fillna(0.0)
    return tab


def typicality(tab):
    """d(v,f), D(v), p(v) = 1/D(v)."""
    d = pd.DataFrame(index=tab.index)
    for f in tab.columns:
        L = float(tab[f].max() - tab[f].min())
        mode = float(tab[f].value_counts().idxmax())
        d[f] = 0.0 if L <= 0 else (tab[f] - mode).abs() / L
    D = d.sum(axis=1)
    p = 1.0 / D.clip(lower=1e-9)
    return p


def pick_method1(users, rng):
    k = max(2, int(round(len(users) * 0.01)))
    return rng.choice(users, size=k, replace=False)


def pick_method2(p, rng):
    a = np.quantile(p.to_numpy(), 0.90)
    cand = p[p > a].index.to_numpy()
    k = max(2, int(round(len(cand) / 10)))
    return rng.choice(cand, size=min(k, len(cand)), replace=False)


def pick_method3(p, rng):
    """p_norm нормируется так, что среднее по пользователям = 0.01;
    пользователь попадает в подвыборку независимо с вероятностью p_norm."""
    users = p.index.to_numpy()
    pn = p / p.mean() * 0.01
    sel = users[rng.random(len(users)) < pn.to_numpy()]
    if len(sel) < 2:                      # страховка от пустой подвыборки
        sel = users[np.argsort(-pn.to_numpy())[:2]]
    return sel


def count_positives(tx, users, horizon=7):
    """Число положительных пар (prior-пара, купленная в горизонте)."""
    sub = tx[tx["visitor"].isin(users)]
    prior = sub[sub["ts"] < T0][["visitor", "item"]].drop_duplicates()
    fut = sub[(sub["ts"] >= T0) &
              (sub["ts"] < T0 + pd.Timedelta(days=horizon))][["visitor", "item"]]
    fut = fut.drop_duplicates()
    return len(prior.merge(fut, on=["visitor", "item"], how="inner"))


def ensure_positives(tx, sel, pool_users, min_pos=10):
    """Дополняет подвыборку типичными пользователями, пока в ней не будет
    min_pos положительных пар.

    Причина: «типичные» по формуле расстояния до моды пользователи в retail --
    в основном однодневные покупатели, у которых после snapshot уже нет
    покупок.  Без положительных примеров классификатор обучить нельзя, поэтому
    подвыборка расширяется пользователями из того же типичного пула (в порядке
    убывания p(v)).  Расширение затрагивает только этап поиска признаков и
    параметров; финальная модель обучается на полном датасете.
    """
    sel = list(sel)
    have = count_positives(tx, sel)
    if have >= min_pos:
        return np.array(sel), 0
    chosen = set(sel)
    added = 0
    for u in pool_users:
        u = int(u)
        if u in chosen:
            continue
        sel.append(u)
        chosen.add(u)
        added += 1
        if count_positives(tx, sel) >= min_pos:
            break
    return np.array(sel), added


# ------------------------------------------------------- short_stochastic_search
def short_search(ev, seed=0):
    keys = list(GPARAM.values())
    p, *_ = stochastic_adaptive(ev, FEATURES, keys, seed=seed,
                                note_prefix="short exp1")
    params = mean_params()
    params.update({k: int(p[k]) for k in keys})
    sig, base_roc, base_pr, _ = exp2_filter(ev, params)
    significant = [f for f in FEATURES if sig[f]] or list(FEATURES)
    p2, *_ = stochastic_adaptive(ev, significant,
                                 ["f:" + f for f in significant],
                                 seed=seed + 1, note_prefix="short exp3")
    opt = opt_from_params(p2, significant)
    roc, pr = ev.score(params_from_opt(opt, significant), significant,
                       note="FINAL subset")
    return significant, opt, roc, pr


def main():
    tx = load_tx()
    full_ds = build_candidates(tx, horizon=7)
    full_tr, full_te = split_dataset(full_ds)
    full_fb = FeatureBuilder(tx, full_ds)
    full_fb.horizon = 7
    users = np.sort(tx["visitor"].unique())
    print(f"full: {len(full_ds):,} pairs, users={len(users):,}")

    tab = user_feature_table(tx)
    p = typicality(tab)
    print(f"typicality: users={len(p)}, p>q90: {int((p > np.quantile(p, 0.9)).sum())}")

    typical_pool = p[p > np.quantile(p, 0.9)].sort_values(ascending=False).index.to_numpy()

    methods = {
        1: ("uniform_1pct", lambda rng: pick_method1(users, rng), users),
        2: ("typical_top10_uniform", lambda rng: pick_method2(p, rng), typical_pool),
        3: ("typical_weighted", lambda rng: pick_method3(p, rng), typical_pool),
    }
    rows, models_log = [], []
    for m, (mname, sampler, pool) in methods.items():
        for s in range(3):
            rng = np.random.default_rng(100 * m + s)
            sel = sampler(rng)
            sel, n_added = ensure_positives(tx, sel, pool)
            txs = tx[tx["visitor"].isin(sel)]
            ds = build_candidates(txs, horizon=7)
            print(f"  method {m} ({mname}) seed {s}: users={len(sel)} "
                  f"(+{n_added}), rows={len(txs):,}, pairs={len(ds):,}, "
                  f"pos={int(ds.label.sum())}")
            if len(ds) < 20 or ds.label.nunique() < 2 or int(ds.label.sum()) < 5:
                print("    слишком мало данных, пропуск")
                continue
            tr, te = split_dataset(ds)
            fb = FeatureBuilder(txs, ds)
            fb.horizon = 7
            ev = Evaluator(fb, tr, te, f"subset_m{m}_{s}", models_log)
            ev.horizon = 7
            sig, opt, roc_s, pr_s = short_search(ev, seed=1000 + 10 * m + s)
            # обучение на полном датасете с найденными признаками/параметрами
            evf = Evaluator(full_fb, full_tr, full_te, f"full_from_m{m}_{s}",
                            models_log)
            evf.horizon = 7
            pfin = params_from_opt(opt, sig)
            roc_f, pr_f = evf.score(pfin, sig, note="FULL dataset")
            name = f"subset_m{m}_{s}"
            with open(os.path.join(MODELS, name + ".txt"), "w", encoding="utf-8") as fh:
                fh.write(evf.last_model.booster_.model_to_string())
            with open(os.path.join(MODELS, name + ".json"), "w", encoding="utf-8") as fh:
                json.dump({"model": name, "method": mname, "seed": s,
                           "n_users": int(len(sel)), "n_added": int(n_added),
                           "features": sig,
                           "windows": {f: int(opt[f]) for f in sig},
                           "roc": round(roc_f, 6), "pr": round(pr_f, 6),
                           "horizon": 7}, fh, ensure_ascii=False, indent=2)
            rows.append({
                "method": m, "method_name": mname, "seed": s,
                "n_users": len(sel), "n_added": n_added, "n_pairs": len(ds),
                "subset_roc": round(roc_s, 6), "subset_pr": round(pr_s, 6),
                "n_features": len(sig),
                "features": "; ".join(f"{f} ({GPARAM[FEATURE_GROUP[f]]}={opt[f]})"
                                      for f in sig),
                "params": json.dumps({("f:" + f): int(opt[f]) for f in sig},
                                     ensure_ascii=False),
                "roc": round(roc_f, 6), "pr": round(pr_f, 6),
            })
            save_csv(rows, "subset_models.csv")
            save_csv(models_log, "subset_models_details.csv")

    df = pd.DataFrame(rows)
    if len(df):
        print("\n=== subset_models ===")
        print(df[["method", "seed", "n_users", "n_added", "n_pairs", "n_features",
                  "subset_roc", "subset_pr", "roc", "pr"]].to_string(index=False))
    print("saved:", os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "results", "subset_models.csv"))


if __name__ == "__main__":
    main()
