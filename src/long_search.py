# -*- coding: utf-8 -*-
"""promt_2: алгоритм long_grid_search.

Итеративное чередование сеточного поиска (с сужающимся вдвое интервалом)
и фильтрации признаков.  Строки таблицы -- признаки, столбцы -- эксперименты.
Метки: 2 -- значимый и изменяемый, 1 -- значимый и неизменяемый, 0 -- незначимый.
В ячейке также указывается текущее оптимальное значение окна признака.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd

from common import (FEATURES, FEATURE_GROUP, GPARAM, GROUPS, MEAN_W, MODELS,
                    RESULTS, WMIN, Y, Evaluator, FeatureBuilder, build_candidates,
                    domain_grid, load_tx, mean_params, save_csv, split_dataset,
                    worse)
from experiments import save_final_model, stochastic_adaptive

def _grid_values(center, length, k=10):
    lo = max(WMIN, center - length / 2)
    hi = min(Y, center + length / 2)
    vals = np.linspace(lo, hi, k)
    return sorted({max(WMIN, min(Y, int(round(v)))) for v in vals})


def _params_from_windows(windows, features):
    return {("f:" + f): int(windows[f]) for f in features}


def run_long(fb, tr, te, horizon=7, verbose=True):
    ev = Evaluator(fb, tr, te, "long_grid_search")
    ev.horizon = horizon
    sig = {f: True for f in FEATURES}
    chg = {f: True for f in FEATURES}
    win = {f: MEAN_W for f in FEATURES}
    group_opt = mean_params()
    columns = []                        # (name, {feature: (mark, window)})
    fits = []

    def marks():
        out = {}
        for f in FEATURES:
            if not sig[f]:
                out[f] = 0
            elif chg[f]:
                out[f] = 2
            else:
                out[f] = 1
        return out

    def record(name):
        columns.append((name, {f: (marks()[f], int(win[f])) for f in FEATURES}))

    def current_params(features=None):
        p = mean_params()
        p.update(_params_from_windows(win, features or [f for f in FEATURES if sig[f]]))
        return p

    def sig_features():
        return [f for f in FEATURES if sig[f]]

    def guard_sig(context=""):
        """Модель без признаков невозможна: если фильтрация удалила все
        признаки, оставляем все (вырожденный случай на крошечных данных)."""
        if not sig_features():
            for f in FEATURES:
                sig[f] = True
            print(f"    [{context}] все признаки незначимы -> оставлены все")

    # ------------------------------------------------ exp1: сеточный поиск по группам
    best_g = {}
    for g, feats in GROUPS.items():
        pname = GPARAM[g]
        trials = []
        for v in domain_grid(10):
            p = mean_params()
            p[pname] = v
            roc, pr = ev.score(p, note=f"exp1[{g}] {pname}={v}")
            trials.append((roc, pr, v))
        roc, pr, v = max(trials, key=lambda x: (x[0], x[1]))
        best_g[g] = v
        fits.extend(trials)
        for f in feats:
            win[f] = v
    group_opt.update({GPARAM[g]: v for g, v in best_g.items()})
    record("exp1")

    # ------------------------------------------------ exp2: фильтрация по группам
    base_roc, base_pr = ev.score(current_params(sig_features()), sig_features(),
                                  note="exp2 baseline")
    for g, feats in GROUPS.items():
        keep = [f for f in FEATURES if sig[f] and f not in feats]
        if not keep:
            for f in feats:
                sig[f] = True
            continue
        roc, pr = ev.score(current_params(), keep, note=f"exp2 drop group [{g}]")
        bad = worse(base_roc, base_pr, roc, pr)
        for f in feats:
            if bad:
                sig[f] = True
            else:
                sig[f] = False
    guard_sig("exp2")
    record("exp2")

    # ------------------------------------------------ exp3: сеточный поиск по признакам
    exp3_rows = []
    for f in [x for x in FEATURES if sig[x]]:
        trials = []
        for v in domain_grid(10):
            p = mean_params()
            p["f:" + f] = v
            roc, pr = ev.score(p, [x for x in FEATURES if sig[x]],
                               note=f"exp3[{f}] f={v}")
            trials.append((roc, pr, v))
        roc, pr, v = max(trials, key=lambda x: (x[0], x[1]))
        base = ev.score(mean_params(), [x for x in FEATURES if sig[x]],
                        note=f"exp3[{f}] baseline mean")
        better = ev.improved(roc, pr, base)
        if better:
            win[f] = v
            chg[f] = True
        else:
            win[f] = MEAN_W
            chg[f] = False
        exp3_rows.append({"feature": f, "group": FEATURE_GROUP[f],
                          "param": GPARAM[FEATURE_GROUP[f]],
                          "window": int(win[f]),
                          "grid_best_roc": round(roc, 6),
                          "grid_best_pr": round(pr, 6),
                          "changeable": int(chg[f]),
                          "significant": int(sig[f])})
    save_csv(exp3_rows, "long_alg_exp_3.csv")
    record("exp3")

    # ------------------------------------------------ exp4: фильтрация по признакам
    def filter_features(note):
        base_roc, base_pr = ev.score(current_params(sig_features()), sig_features(),
                                      note=f"{note} baseline")
        sig_feats = [f for f in FEATURES if sig[f]]
        for f in sig_feats:
            keep = [x for x in sig_feats if x != f]
            if not keep:
                continue
            roc, pr = ev.score(current_params(keep), keep,
                               note=f"{note} drop [{f}]")
            if not worse(base_roc, base_pr, roc, pr):
                sig[f] = False

    filter_features("exp4")
    guard_sig("exp4")
    record("exp4")

    # ------------------------------------------------ итерации: поиск + фильтрация
    n0 = float(Y - WMIN)          # изначальная длина промежутка
    prev_best = ev.score(current_params(sig_features()), sig_features(),
                          note="iter0 model")
    no_improve = 0
    k = 0
    while k < 15:
        k += 1
        # сеточный поиск-k: длина промежутка n / 2^(k-1)
        length = n0 / (2.0 ** (k - 1))
        if length < 10:
            break
        # сеточный поиск по изменяемым значимым признакам
        for f in [x for x in FEATURES if sig[x] and chg[x]]:
            vals = _grid_values(win[f], length, 10)
            trials = []
            for v in vals:
                p = current_params()
                p["f:" + f] = v
                roc, pr = ev.score(p, sig_features(),
                                   note=f"iter{k} grid[{f}] f={v}")
                trials.append((roc, pr, v))
            if not trials:
                continue
            roc, pr, v = max(trials, key=lambda x: (x[0], x[1]))
            ref = ev.score(current_params(sig_features()), sig_features(),
                           note=f"iter{k} ref[{f}]")
            if ev.improved(roc, pr, ref):
                win[f] = v
            else:
                chg[f] = False          # параметр больше не влияет -> фиксируем
        record(f"iter{k}_grid")
        # фильтрация
        filter_features(f"iter{k}_filter")
        guard_sig(f"iter{k}_filter")
        record(f"iter{k}_filter")
        cur = ev.score(current_params(sig_features()), sig_features(),
                        note=f"iter{k} model")
        if ev.improved(cur[0], cur[1], prev_best):
            prev_best = cur
            no_improve = 0
        else:
            no_improve += 1
        if verbose:
            print(f"  iter {k}: length={length:.1f} roc={cur[0]:.4f} pr={cur[1]:.4f} "
                  f"sig={sum(sig.values())} chg={sum(1 for f in FEATURES if sig[f] and chg[f])} "
                  f"no_improve={no_improve}")
        if no_improve >= 3:
            break

    # ------------------------------------------------ таблица
    table = {"feature": FEATURES, "group": [FEATURE_GROUP[f] for f in FEATURES]}
    for name, m in columns:
        table[name] = [f"{m[f][0]} (w={m[f][1]})" for f in FEATURES]
    df = pd.DataFrame(table)
    df.to_csv(os.path.join(RESULTS, "long_grid_search.csv"), index=False,
              sep=";", encoding="utf-8-sig")

    # ------------------------------------------------ финал
    significant = [f for f in FEATURES if sig[f]]
    roc, pr = ev.score(current_params(significant), significant,
                       note="FINAL long_grid_search")
    save_final_model(ev, significant, {f: win[f] for f in significant},
                     "long_grid_search_res_final")
    cfg = {"features": significant,
           "windows": {f: int(win[f]) for f in significant},
           "params": {f: GPARAM[FEATURE_GROUP[f]] for f in significant},
           "roc": round(roc, 6), "pr": round(pr, 6),
           "iterations": k, "horizon": horizon}
    with open(os.path.join(MODELS, "long_grid_search_res_config.json"), "w",
              encoding="utf-8") as fh:
        json.dump(cfg, fh, ensure_ascii=False, indent=2)

    save_csv(ev.log, "long_grid_search_res_models.csv")
    pd.DataFrame([{"feature": f, "mark": marks()[f], "window": int(win[f])}
                  for f in FEATURES]).to_csv(
        os.path.join(RESULTS, "long_grid_search_res_features.csv"),
        index=False, sep=";", encoding="utf-8-sig")
    return {"features": significant, "windows": {f: win[f] for f in significant},
            "roc": round(roc, 6), "pr": round(pr, 6), "iterations": k}


def main():
    tx = load_tx()
    ds = build_candidates(tx, horizon=7)
    tr, te = split_dataset(ds)
    fb = FeatureBuilder(tx, ds)
    fb.horizon = 7
    print(f"long_grid_search: pairs={len(ds):,}")
    res = run_long(fb, tr, te, horizon=7)
    print("FINAL:", res["roc"], res["pr"], "features:", len(res["features"]),
          "iterations:", res["iterations"])


if __name__ == "__main__":
    main()
