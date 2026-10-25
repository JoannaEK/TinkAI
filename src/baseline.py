# -*- coding: utf-8 -*-
"""Опорная модель: все 19 признаков, все окна = среднее (363), без отбора.

Нужна как точка отсчёта: показывает, сколько даёт подбор окон/признаков
по сравнению с «никакой настройки».  Результат -- results/baseline_mean.csv.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd

from common import (Evaluator, FeatureBuilder, RESULTS, build_candidates,
                    load_tx, mean_params, save_csv, split_dataset)


def main():
    tx = load_tx()
    rows = []
    for h in (7, 14):
        ds = build_candidates(tx, horizon=h)
        tr, te = split_dataset(ds)
        ev = Evaluator(FeatureBuilder(tx, ds), tr, te, f"baseline_h{h}")
        roc, pr = ev.score(mean_params(), note="baseline: 19 features, mean window")
        rows.append({"horizon": h, "n_features": 19, "params": "all=363",
                     "n_pairs": len(ds), "n_pos": int(ds.label.sum()),
                     "roc": round(roc, 6), "pr": round(pr, 6)})
        print(f"h={h}: ROC={roc:.6f} PR={pr:.6f}")
    save_csv(pd.DataFrame(rows), os.path.join(RESULTS, "baseline_mean.csv"))


if __name__ == "__main__":
    main()
