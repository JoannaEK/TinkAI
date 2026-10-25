# -*- coding: utf-8 -*-
"""Эксперименты 1-3, финал и их варианты для promt_0/promt_1."""
import json
import os
import time

import numpy as np
import pandas as pd

from common import (FEATURES, FEATURE_GROUP, GPARAM, GROUPS, MEAN_W, MODELS,
                    RESULTS, Evaluator, domain_grid, mean_params,
                    resolve_windows, save_csv, worse, Y, WMIN)


# --------------------------------------------------------------- эксперименты
def exp1_grid(ev, k=5, tag="", log=None):
    """Сеточный поиск по каждой группе: k значений, остальные параметры -- mean."""
    best, rows = {}, []
    for g, feats in GROUPS.items():
        pname = GPARAM[g]
        trials = []
        for v in domain_grid(k):
            p = mean_params()
            p[pname] = v
            roc, pr = ev.score(p, note=f"exp1[{g}] {pname}={v}")
            trials.append((roc, pr, v))
        roc, pr, v = max(trials, key=lambda x: (x[0], x[1]))
        best[g] = v
        for r, p_, vv in trials:
            rows.append({"group": g, "features": "; ".join(feats),
                         "param": pname, "value": vv,
                         "roc": round(r, 6), "pr": round(p_, 6),
                         "selected": int(vv == v)})
    return best, rows


def exp2_filter(ev, params, tag=""):
    """Фильтрация по группам. Возвращает значимость каждого признака.

    Таблица exp_2.csv -- по строке на каждый признак, метка 1/0.
    """
    base_roc, base_pr = ev.score(params, FEATURES,
                                  note="exp2 baseline (all features)")
    sig, rows, groups = {}, [], {}
    for g, feats in GROUPS.items():
        keep = [f for f in FEATURES if f not in feats]
        if not keep:
            for f in feats:
                sig[f] = True
            groups[g] = (base_roc, base_pr, True)
            continue
        roc, pr = ev.score(params, keep, note=f"exp2 drop group [{g}]")
        bad = worse(base_roc, base_pr, roc, pr)
        for f in feats:
            sig[f] = bad
        groups[g] = (roc, pr, bad)
    for g, feats in GROUPS.items():
        roc, pr, bad = groups[g]
        for f in feats:
            rows.append({"feature": f, "group": g, "param": GPARAM[g],
                         "dropped_roc": round(roc, 6), "dropped_pr": round(pr, 6),
                         "significant": int(sig[f])})
    return sig, base_roc, base_pr, rows


def exp3_grid(ev, significant, k=5):
    """Каждому значимому признаку -- свой параметр; k значений, остальные mean."""
    opt, rows = {}, []
    for f in significant:
        key = "f:" + f
        trials = []
        for v in domain_grid(k):
            p = mean_params()
            p[key] = v
            roc, pr = ev.score(p, significant, note=f"exp3[{f}] {key}={v}")
            trials.append((roc, pr, v))
        roc, pr, v = max(trials, key=lambda x: (x[0], x[1]))
        opt[f] = v
        for r, p_, vv in trials:
            rows.append({"feature": f, "param": key, "value": vv,
                         "roc": round(r, 6), "pr": round(p_, 6),
                         "selected": int(vv == v)})
    return opt, rows


def exp4_refine(ev, opt, features, seeds=(-20, -10, 0, 10, 20)):
    """Сеточный поиск с шагом 10 в окрестности оптимума (по каждому признаку)."""
    base = params_from_opt(opt, features)
    for f in list(features):
        key = "f:" + f
        cur = opt.get(f, MEAN_W)
        trials = []
        for d in seeds:
            v = max(WMIN, min(Y, cur + d))
            p = dict(base)
            p[key] = v
            roc, pr = ev.score(p, features, note=f"exp4[{f}] {key}={v}")
            trials.append((roc, pr, v))
        roc, pr, v = max(trials, key=lambda x: (x[0], x[1]))
        opt[f] = v
        base[key] = v
    return opt


def params_from_opt(opt, features):
    p = mean_params()
    for f in features:
        p["f:" + f] = opt.get(f, MEAN_W)
    return p


def final_score(ev, significant, opt, tag, horizon):
    p = params_from_opt(opt, significant)
    roc, pr = ev.score(p, significant, note="FINAL")
    return {"version": tag, "horizon": horizon,
            "features": "; ".join(f"{f} ({GPARAM[FEATURE_GROUP[f]]}={opt[f]})"
                                  for f in significant),
            "params": json.dumps({("f:" + f): int(opt[f]) for f in significant},
                                 ensure_ascii=False),
            "n_features": len(significant),
            "roc": round(roc, 6), "pr": round(pr, 6)}


# --------------------------------------------------------------- стохастика
def stochastic_adaptive(ev, features, keys, start=None, step0=80, step_min=5,
                        patience=None, max_evals=4000, seed=0, note_prefix="stoch"):
    """Стохастический поиск с адаптивным шагом (80 -> 5)."""
    rng = np.random.default_rng(seed)
    p = mean_params()
    if start:
        p.update(start)
    best_roc, best_pr = ev.score(p, features, note=f"{note_prefix} start")
    state = {k: {"step": step0, "tried": set()} for k in keys}
    no_improve = 0
    evals = 1
    while evals < max_evals:
        alive = [k for k in keys if state[k]["step"] >= step_min]
        if not alive:
            break
        k = alive[int(rng.integers(len(alive)))]
        st = state[k]
        step = st["step"]
        dirs = [d for d in (+1, -1) if d not in st["tried"]]
        if not dirs:
            st["step"] = step // 2
            st["tried"] = set()
            continue
        d = dirs[int(rng.integers(len(dirs)))]
        st["tried"].add(d)
        new_val = max(WMIN, min(Y, int(p.get(k, MEAN_W) + d * step)))
        trial = dict(p)
        trial[k] = new_val
        roc, pr = ev.score(trial, features,
                           note=f"{note_prefix} {k}:{p.get(k, MEAN_W)}->{new_val} (step {step})")
        evals += 1
        if ev.improved(roc, pr, (best_roc, best_pr)):
            p, best_roc, best_pr = trial, roc, pr
            no_improve = 0
        else:
            no_improve += 1
            if patience is not None and no_improve >= patience:
                break
    return p, best_roc, best_pr


def stochastic_fixed(ev, features, keys, step=10, patience=10, max_evals=4000,
                     seed=0, note_prefix="stoch_fixed"):
    rng = np.random.default_rng(seed)
    p = mean_params()
    best_roc, best_pr = ev.score(p, features, note=f"{note_prefix} start")
    no_improve = 0
    evals = 1
    while evals < max_evals and no_improve < patience:
        k = keys[int(rng.integers(len(keys)))]
        d = 1 if rng.random() < 0.5 else -1
        new_val = max(WMIN, min(Y, int(p.get(k, MEAN_W) + d * step)))
        trial = dict(p)
        trial[k] = new_val
        roc, pr = ev.score(trial, features,
                           note=f"{note_prefix} {k}:{p.get(k, MEAN_W)}->{new_val}")
        evals += 1
        if ev.improved(roc, pr, (best_roc, best_pr)):
            p, best_roc, best_pr = trial, roc, pr
            no_improve = 0
        else:
            no_improve += 1
    return p, best_roc, best_pr


def opt_from_params(p, features):
    return {f: int(p.get("f:" + f, p[GPARAM[FEATURE_GROUP[f]]])) for f in features}


# --------------------------------------------------------------- варианты
def save_final_model(ev, significant, opt, name):
    p = params_from_opt(opt, significant)
    X, feats = ev.fb.matrix(p, significant)
    y = ev.fb.cand["label"].to_numpy()
    import lightgbm as lgb
    from common import LGB_PARAMS
    m = lgb.LGBMClassifier(**LGB_PARAMS)
    m.fit(X[ev.tr], y[ev.tr])
    path = os.path.join(MODELS, name + ".txt")
    # LightGBM не умеет писать в пути с не-ASCII символами -> пишем через Python
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(m.booster_.model_to_string())
    cfg = {"model": name, "features": significant,
           "windows": {("f:" + f): int(opt[f]) for f in significant},
           "params": {k: int(v) for k, v in p.items()},
           "lgb_params": LGB_PARAMS, "snapshot": str(ev.fb.t),
           "horizon": int(ev.horizon) if hasattr(ev, "horizon") else None}
    with open(os.path.join(MODELS, name + ".json"), "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, ensure_ascii=False, indent=2)
    return path


def _sig_list(sig):
    """Список значимых признаков; если фильтрация удалила все -- оставляем все
    (модель без признаков невозможна, на крошечных подвыборках это бывает)."""
    out = [f for f in FEATURES if sig[f]]
    return out if out else list(FEATURES)


def run_variant(ev, version, horizon, mode, k=5, seed=0, save_v1_tables=False):
    """mode: 'grid5' | 'grid10' | 'grid10_exp4' | 'stoch_fixed10' |
             'stoch_adaptive' | 'mean_then_exp3' | 'perfeature_adaptive'"""
    t0 = time.time()
    tables = {}
    kk = None
    if mode in ("grid5", "grid10", "grid10_exp4"):
        kk = 5 if mode == "grid5" else 10
        best_g, exp1_rows = exp1_grid(ev, k=kk)
        params = mean_params()
        params.update({GPARAM[g]: v for g, v in best_g.items()})
        sig, base_roc, base_pr, exp2_rows = exp2_filter(ev, params)
        significant = _sig_list(sig)
        opt, exp3_rows = exp3_grid(ev, significant, k=kk)
        if mode == "grid10_exp4":
            opt = exp4_refine(ev, opt, significant)
        tables = {"exp1": exp1_rows, "exp2": exp2_rows, "exp3": exp3_rows}
    elif mode in ("stoch_fixed10", "stoch_adaptive"):
        keys = list(GPARAM.values())
        if mode == "stoch_fixed10":
            p, *_ = stochastic_fixed(ev, FEATURES, keys, step=10, patience=10,
                                     seed=seed, note_prefix=f"exp1[{version}]")
        else:
            p, *_ = stochastic_adaptive(ev, FEATURES, keys, seed=seed,
                                        note_prefix=f"exp1[{version}]")
        best_g = {GPARAM[g]: int(p[GPARAM[g]]) for g in GROUPS}
        params = mean_params()
        params.update(best_g)
        sig, base_roc, base_pr, exp2_rows = exp2_filter(ev, params)
        significant = _sig_list(sig)
        keys_f = ["f:" + f for f in significant]
        if mode == "stoch_fixed10":
            p2, *_ = stochastic_fixed(ev, significant, keys_f, step=10,
                                      patience=10, seed=seed + 1,
                                      note_prefix=f"exp3[{version}]")
        else:
            p2, *_ = stochastic_adaptive(ev, significant, keys_f,
                                         seed=seed + 1,
                                         note_prefix=f"exp3[{version}]")
        opt = opt_from_params(p2, significant)
        tables = {"exp2": exp2_rows}
    elif mode == "mean_then_exp3":
        params = mean_params()
        sig, base_roc, base_pr, exp2_rows = exp2_filter(ev, params)
        significant = _sig_list(sig)
        p2, *_ = stochastic_adaptive(ev, significant,
                                     ["f:" + f for f in significant],
                                     seed=seed, note_prefix=f"exp3[{version}]")
        opt = opt_from_params(p2, significant)
        tables = {"exp2": exp2_rows}
    elif mode == "perfeature_adaptive":
        p, roc, pr = stochastic_adaptive(ev, FEATURES,
                                         ["f:" + f for f in FEATURES],
                                         patience=20, seed=seed,
                                         note_prefix=f"stoch[{version}]")
        significant = list(FEATURES)
        opt = opt_from_params(p, significant)
        tables = {}
    else:
        raise ValueError(mode)

    final = final_score(ev, significant, opt, version, horizon)
    final["mode"] = mode
    final["seconds"] = round(time.time() - t0, 1)
    if kk is not None:
        final["k"] = kk
    save_final_model(ev, significant, opt, f"final_{version}")
    return final, tables

