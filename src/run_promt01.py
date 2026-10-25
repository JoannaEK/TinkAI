# -*- coding: utf-8 -*-
"""promt_0 + promt_1: базовый алгоритм и все его варианты.

Результаты: results/final_models_diff_versions.csv, results/models.csv,
results/exp_1.csv, exp_2.csv, exp_3.csv (для базового варианта) и
results/experiments/*.csv для остальных вариантов.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd

from common import (FeatureBuilder, Evaluator, RESULTS, build_candidates,
                    load_tx, save_csv, split_dataset)
from experiments import run_variant

VERSIONS = [
    ("v1_grid5_h7",          7,  "grid5",              5),
    ("v2_grid5_h14",         14, "grid5",              5),
    ("v3_grid10_h14",        14, "grid10",             10),
    ("v4_grid10_exp4_h14",   14, "grid10_exp4",        10),
    ("v5_stoch_fixed10_h14", 14, "stoch_fixed10",      10),
    ("v6_stoch_adaptive_h14", 14, "stoch_adaptive",    10),
    ("v7_mean_exp3_h14",     14, "mean_then_exp3",     10),
    ("v8_perfeature_h14",    14, "perfeature_adaptive", 10),
]


def main():
    only = os.environ.get("VERSIONS", "").strip()
    wanted = set(only.split(",")) if only else None
    quick = int(os.environ.get("QUICK_USERS", "0"))
    tx = load_tx(quick_users=quick or None)
    print(f"tx: {len(tx):,} rows, users={tx.visitor.nunique():,}")

    datasets = {}
    for h in (7, 14):
        ds = build_candidates(tx, horizon=h)
        tr, te = split_dataset(ds)
        fb = FeatureBuilder(tx, ds)
        fb.horizon = h
        datasets[h] = (ds, tr, te, fb)
        print(f"dataset h={h}: pairs={len(ds):,}, pos={int(ds.label.sum()):,}")

    models_log = []
    finals = []
    expdir = os.path.join(RESULTS, "experiments")
    os.makedirs(expdir, exist_ok=True)

    for version, h, mode, k in VERSIONS:
        if wanted and version not in wanted:
            continue
        ds, tr, te, fb = datasets[h]
        ev = Evaluator(fb, tr, te, version, models_log)
        ev.horizon = h
        t0 = time.time()
        print(f"\n=== {version} ({mode}, k={k}, horizon={h}) ===")
        final, tables = run_variant(ev, version, h, mode, k=k)
        finals.append(final)
        print(f"  final ROC={final['roc']:.4f} PR={final['pr']:.4f} "
              f"({time.time()-t0:.1f}s, fits={ev.n_fit})")

        for name, rows in tables.items():
            if not rows:
                continue
            base_name = {"exp1": "exp_1", "exp2": "exp_2", "exp3": "exp_3"}.get(name, name)
            if version == "v1_grid5_h7":
                save_csv(rows, f"{base_name}.csv")
            else:
                save_csv(rows, f"experiments/{base_name}_{version}.csv")

        # для стохастических вариантов отдельные таблицы строятся по логу прогонов
        derived = {}
        for prefix, fname in (("exp1", "exp_1"), ("stoch", "exp_1"),
                              ("exp3", "exp_3")):
            key = f"{prefix}[{version}]"
            rows = [r for r in models_log if r["note"].startswith(key)]
            if rows:
                derived.setdefault(fname, []).extend(rows)
        for fname, rows in derived.items():
            out_name = (f"{fname}.csv" if version == "v1_grid5_h7"
                        else f"experiments/{fname}_{version}.csv")
            if not os.path.exists(os.path.join(RESULTS, out_name)):
                save_csv(rows, out_name)

        save_csv(models_log, "models.csv")
        save_csv(finals, "final_models_diff_versions.csv")

    # сводка
    df = pd.DataFrame(finals)
    if len(df):
        print("\n=== ФИНАЛЬНЫЕ МОДЕЛИ ===")
        print(df[["version", "horizon", "mode", "n_features", "roc", "pr"]]
              .to_string(index=False))
    print("\nresults:", RESULTS)


if __name__ == "__main__":
    main()
