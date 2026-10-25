# -*- coding: utf-8 -*-
"""Сводное сравнение всех обученных моделей.

Строит results/COMPARISON.md: этапы поиска по каждому варианту, вклад
групп признаков (эксперимент 2), доверительные интервалы ROC AUC
(Hanley-McNeil), таблицы long_grid_search и subset-моделей.

Запуск: py tools/compare_models.py
"""
import math
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(RES, "COMPARISON.md")

# число объектов в тесте (horizon -> (positives, negatives))
TEST = {7: (1367, 5466), 14: (2505, 10021)}


def se_auc(auc, n1, n0):
    """Стандартная ошибка AUC по Hanley & McNeil (1982)."""
    q1 = auc / (2.0 - auc)
    q2 = 2.0 * auc * auc / (1.0 + auc)
    var = (auc * (1 - auc) + (n1 - 1) * (q1 - auc * auc) +
           (n0 - 1) * (q2 - auc * auc)) / (n1 * n0)
    return math.sqrt(max(var, 0.0))


def ci(auc, horizon):
    n1, n0 = TEST[horizon]
    s = se_auc(auc, n1, n0)
    return s, 1.96 * s


def read(name):
    path = os.path.join(RES, name)
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, sep=";")


def baseline_of(models, variant):
    """Базовая модель эксперимента 2.

    У сеточных вариантов она логируется (19 признаков).  У стохастических
    конфигурация совпадает с последней точкой exp1 и попадает в кэш
    Evaluator, поэтому строки в логе нет -- берём лучший прогон exp1.
    """
    g = models[(models["variant"] == variant) &
               (models["note"].str.startswith("exp2"))]
    full = g[g["n_features"] == 19]
    if len(full):
        return full.iloc[0]
    exp1 = models[(models["variant"] == variant) &
                  (models["note"].str.startswith("exp1"))]
    if len(exp1):
        return exp1.loc[exp1["roc"].idxmax()]
    return None


def stage_rows(models):
    """Лучший ROC на каждом этапе поиска для каждого варианта."""
    rows = []
    for variant, g in models.groupby("variant", sort=False):
        exp1 = g[g["note"].str.startswith("exp1")]
        exp2 = g[g["note"].str.startswith("exp2")]
        exp3 = g[g["note"].str.startswith("exp3")]
        exp4 = g[g["note"].str.startswith("exp4")]
        base = baseline_of(models, variant)
        rows.append({
            "variant": variant,
            "n_fits": len(g),
            "exp1_best_roc": exp1["roc"].max() if len(exp1) else None,
            "exp1_best_pr": exp1["pr"].max() if len(exp1) else None,
            "exp2_base_roc": base["roc"] if base is not None else None,
            "exp2_base_pr": base["pr"] if base is not None else None,
            "exp2_base_nfeat": 19 if base is not None else None,
            "exp3_best_roc": exp3["roc"].max() if len(exp3) else None,
            "exp3_best_pr": exp3["pr"].max() if len(exp3) else None,
            "exp4_best_roc": exp4["roc"].max() if len(exp4) else None,
        })
    return pd.DataFrame(rows)


def exp2_rows(models, variant):
    g = models[(models["variant"] == variant) &
               (models["note"].str.startswith("exp2"))]
    if not len(g):
        return None
    return g


def worse(base_roc, base_pr, roc, pr):
    if base_roc - roc >= 0.01:
        return True
    if abs(base_roc - roc) < 0.01 and base_pr - pr >= 0.005:
        return True
    return False


def main():
    models = read("models.csv")
    finals = read("final_models_diff_versions.csv")
    long_models = read("long_grid_search_res_models.csv")
    subsets = read("subset_models.csv")
    if models is None or finals is None:
        print("нет results/models.csv или final_models_diff_versions.csv")
        return 1

    hz = dict(zip(finals["version"], finals["horizon"]))
    md = ["# Сводное сравнение алгоритмов", ""]
    md.append("Сгенерировано `tools/compare_models.py` по "
              "`results/models.csv`, `final_models_diff_versions.csv`, "
              "`long_grid_search_res_models.csv`, `subset_models.csv`.")
    md.append("")

    # ---------------------------------------------------------- финалы
    md.append("## 1. Финальные модели promt_0 / promt_1")
    md.append("")
    md.append("| версия | horizon | поиск | признаков | ROC | ±95% CI | PR |")
    md.append("|---|---|---|---|---|---|---|")
    for _, r in finals.iterrows():
        h = int(hz.get(r["version"], 7))
        s, d = ci(float(r["roc"]), h)
        md.append(f"| {r['version']} | {h} | {r.get('mode', '')} | "
                  f"{int(r['n_features'])} | {r['roc']:.6f} | "
                  f"{d:.5f} | {r['pr']:.6f} |")
    md.append("")

    # ---------------------------------------------------------- этапы
    st = stage_rows(models)
    md.append("## 2. Качество по этапам поиска (ROC, лучшая модель этапа)")
    md.append("")
    md.append("| версия | обучений | exp1 (все 19) | exp2 базовая | exp3 (свои окна) | exp4 (±20) | финал |")
    md.append("|---|---|---|---|---|---|---|")
    for _, r in st.iterrows():
        def f(x):
            return "—" if pd.isna(x) else f"{x:.4f}"
        fin = finals[finals["version"] == r["variant"]]
        froc = float(fin.iloc[0]["roc"]) if len(fin) else None
        md.append(f"| {r['variant']} | {r['n_fits']} | {f(r['exp1_best_roc'])} | "
                  f"{f(r['exp2_base_roc'])} | {f(r['exp3_best_roc'])} | "
                  f"{f(r['exp4_best_roc'])} | {f(froc)} |")
    md.append("")

    # ---------------------------------------------------------- вклад групп
    md.append("## 3. Вклад групп признаков (эксперимент 2, ΔROC/ΔPR к базовой модели)")
    md.append("")
    for variant in models["variant"].drop_duplicates():
        g = exp2_rows(models, variant)
        if g is None:
            continue
        base = baseline_of(models, variant)
        if base is None:
            continue
        md.append(f"### {variant}")
        md.append("")
        md.append("| удалённая группа | ROC | ΔROC | PR | ΔPR | хуже? |")
        md.append("|---|---|---|---|---|---|")
        for _, r in g.iterrows():
            if r["n_features"] >= 19:
                continue
            note = str(r["note"])
            grp = note.split("[")[1].rstrip("]") if "[" in note else note
            dra = r["roc"] - base["roc"]
            drp = r["pr"] - base["pr"]
            flag = "да" if worse(base["roc"], base["pr"], r["roc"], r["pr"]) else "нет"
            md.append(f"| {grp} | {r['roc']:.6f} | {dra:+.6f} | "
                      f"{r['pr']:.6f} | {drp:+.6f} | {flag} |")
        md.append("")

    # ---------------------------------------------------------- long
    if long_models is not None:
        md.append("## 4. promt_2: long_grid_search")
        md.append("")
        md.append("| шаг | признаков | ROC | PR |")
        md.append("|---|---|---|---|")
        picks = []
        for label, mask in [
            ("exp1 (сетка 10, все 19)", long_models["note"].str.startswith("exp1")),
            ("exp2 (фильтрация групп)", long_models["note"] == "exp2 baseline"),
            ("exp3 (свои окна)", long_models["note"].str.startswith("exp3")),
            ("exp4 (фильтрация признаков)", long_models["note"] == "exp4 baseline"),
        ]:
            g = long_models[mask]
            if len(g):
                r = g.loc[g["roc"].idxmax()]
                picks.append((label, r))
        n_iter = 0
        for i in range(1, 12):
            g = long_models[long_models["note"].str.startswith(f"iter{i} grid")]
            if not len(g):
                break
            n_iter = i
            picks.append((f"iter{i} (сетка)", g.loc[g["roc"].idxmax()]))
        for label, r in picks:
            md.append(f"| {label} | {int(r['n_features'])} | "
                      f"{r['roc']:.6f} | {r['pr']:.6f} |")
        # финальные метрики берём из лога прогона
        log_path = os.path.join(RES, "promt02_run.log")
        if os.path.exists(log_path):
            raw = open(log_path, "rb").read()
            for enc in ("utf-8", "utf-16"):
                try:
                    text = raw.decode(enc)
                except UnicodeDecodeError:
                    continue
                if "FINAL:" in text:
                    break
            for line in text.splitlines():
                if line.startswith("FINAL:"):
                    parts = line.split()
                    md.append(f"| финал (iter{n_iter + 1}) | "
                              f"{parts[4]} | {parts[1]} | {parts[2]} |")
        md.append("")

    # ---------------------------------------------------------- subset
    if subsets is not None:
        md.append("## 5. promt_3: модели на подвыборках")
        md.append("")
        md.append("| способ | метод | seed | пользователей | добавлено | пар | признаков | ROC подвыборки | ROC (полный) | ±95% CI | PR (полный) |")
        md.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for _, r in subsets.iterrows():
            s, d = ci(float(r["roc"]), 7)
            md.append(f"| {int(r['method'])} | {r['method_name']} | {int(r['seed'])} | "
                      f"{int(r['n_users'])} | {int(r['n_added'])} | {int(r['n_pairs'])} | "
                      f"{int(r['n_features'])} | {r['subset_roc']:.4f} | "
                      f"{r['roc']:.6f} | {d:.5f} | {r['pr']:.6f} |")
        md.append("")
        agg = subsets.groupby("method")[["n_users", "n_pairs", "n_features", "roc", "pr"]].mean()
        md.append("Средние по способам отбора:")
        md.append("")
        md.append("| способ | пользователей | пар | признаков | ROC | PR |")
        md.append("|---|---|---|---|---|---|")
        for m, r in agg.iterrows():
            md.append(f"| {int(m)} | {r['n_users']:.0f} | {r['n_pairs']:.0f} | "
                      f"{r['n_features']:.1f} | {r['roc']:.4f} | {r['pr']:.4f} |")
        md.append("")

    # ---------------------------------------------------------- важность
    import glob
    import json

    md.append("## 6. Важность признаков (gain, топ-6) финальных моделей")
    md.append("")
    md.append("| версия | вклад признаков, % gain |")
    md.append("|---|---|")
    for path in sorted(glob.glob(os.path.join(ROOT, "models", "final_v*.txt"))):
        name = os.path.basename(path)[len("final_"):-len(".txt")]
        cfg_path = os.path.join(ROOT, "models", f"final_{name}.json")
        if not os.path.exists(cfg_path):
            continue
        cfg = json.load(open(cfg_path, encoding="utf-8"))
        feats = cfg["features"]
        import lightgbm as lgb
        booster = lgb.Booster(model_str=open(path, encoding="utf-8").read())
        gain = booster.feature_importance("gain")
        total = sum(gain) or 1
        top = sorted(zip(feats, gain), key=lambda x: -x[1])[:6]
        md.append(f"| {name} | " +
                  "; ".join(f"{f} {v / total * 100:.0f}%" for f, v in top) + " |")
    md.append("")

    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")
    print("written", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
