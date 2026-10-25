# -*- coding: utf-8 -*-
"""Сборка сводки результатов в markdown (results/RESULTS.md)."""
import json
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results")

FEATURE_PARAM = {
    "user_tx_cnt": "n", "user_avg_active_day_gap": "n", "user_avg_tx": "n",
    "item_tx_cnt": "m", "item_avg_active_day_gap": "m",
    "item_repeat_purchase_rate": "m", "item_avg_tx": "m",
    "cat_tx_cnt": "k", "cat_avg_active_day_gap": "k",
    "cat_repeat_purchase_rate": "k", "cat_avg_tx": "k",
    "vcat_tx_cnt": "x", "vcat_avg_active_day_gap": "x",
    "vcat_repeat_purchase_rate": "x", "vcat_avg_tx": "x",
    "vi_tx_cnt": "q", "vc_tx_cnt": "s", "ic_tx_cnt": "z",
    "ii_co_purchase": "u",
}


def md_table(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |",
           "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        out.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(out)


def main():
    parts = []
    p = os.path.join(R, "final_models_diff_versions.csv")
    if os.path.exists(p):
        d = pd.read_csv(p, sep=";")
        t = d[["version", "horizon", "mode", "n_features", "roc", "pr"]].copy()
        t.columns = ["версия", "horizon", "поиск", "признаков", "ROC", "PR"]
        parts.append("### promt_0 / promt_1: финальные модели\n\n" + md_table(t))
        det = d[["version", "features"]].copy()
        det.columns = ["версия", "признаки (оптимальное окно, дней)"]
        parts.append("Признаки финальных моделей:\n\n" + md_table(det))

    p = os.path.join(R, "baseline_mean.csv")
    if os.path.exists(p):
        d = pd.read_csv(p, sep=";")
        t = d[["horizon", "n_features", "n_pairs", "n_pos", "roc", "pr"]].copy()
        t.columns = ["horizon", "признаков", "пар", "positives", "ROC", "PR"]
        parts.append("### Опорная модель (все 19 признаков, окно 363)\n\n"
                     + md_table(t))

    p = os.path.join(ROOT, "models", "long_grid_search_res_final.json")
    pcfg = os.path.join(ROOT, "models", "long_grid_search_res_config.json")
    if os.path.exists(p) and os.path.exists(pcfg):
        with open(p, encoding="utf-8") as fh:
            cfg = json.load(fh)
        with open(pcfg, encoding="utf-8") as fh:
            meta = json.load(fh)
        t = pd.DataFrame([{"признак": f,
                           "групповой параметр": FEATURE_PARAM.get(f, ""),
                           "оптимальное окно, дней": cfg["windows"].get(
                               "f:" + f, cfg["windows"].get(f))}
                          for f in cfg["features"]])
        parts.append(
            f"### promt_2: long_grid_search\n\n"
            f"ROC = **{meta['roc']:.4f}**, PR = **{meta['pr']:.4f}**, "
            f"итераций: {meta['iterations']}, horizon = {meta['horizon']}.\n\n"
            + md_table(t))

    p = os.path.join(R, "subset_models.csv")
    if os.path.exists(p):
        d = pd.read_csv(p, sep=";")
        d = d[["method", "method_name", "seed", "n_users", "n_added", "n_pairs",
               "n_features", "subset_roc", "subset_pr", "roc", "pr"]]
        d.columns = ["способ", "метод", "выборка", "пользователей", "добавлено",
                     "пар", "признаков", "ROC (подвыборка)", "PR (подвыборка)",
                     "ROC (полный)", "PR (полный)"]
        parts.append("### promt_3: short_stochastic_search на подвыборках\n\n"
                     + md_table(d))

    text = "\n\n".join(parts)
    with open(os.path.join(R, "RESULTS.md"), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
