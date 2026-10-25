# -*- coding: utf-8 -*-
"""Общее ядро проекта RecSyS.

Данные: отредактированная retail-таблица (itemid, Quantity, InvoiceDate,
visitor, visitor_cat, item_cat).  Каждая строка -- транзакция.

Snapshot t фиксирован (T0), горизонт таргета задаётся (7/14 дней).
Кандидаты -- пары (visitor, item), у которых была хотя бы одна транзакция
до t ("prior interaction").  Положительные -- покупка в [t, t+horizon),
отрицательные -- случайная подвыборка остальных prior-пар.
"""
import copy
import json
import math
import os
import time
from collections import OrderedDict

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.environ.get(
    "RETAIL_CSV", os.path.join(ROOT, "data", "online_retail_II_edited.csv"))
RESULTS = os.path.join(ROOT, "results")
MODELS = os.path.join(ROOT, "models")
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(MODELS, exist_ok=True)

T0 = pd.Timestamp("2011-11-20 00:00:00")
FIRST_TS = pd.Timestamp("2009-12-01 00:00:00")
Y = (T0 - FIRST_TS).days                      # 718 дней наблюдения
WMIN = 7
MEAN_W = int(round((WMIN + Y) / 2))           # среднее значение области

# ---------------------------------------------------------------- признаки
GROUPS = OrderedDict([
    ("user",        ["user_tx_cnt", "user_avg_active_day_gap", "user_avg_tx"]),
    ("item",        ["item_tx_cnt", "item_avg_active_day_gap",
                     "item_repeat_purchase_rate", "item_avg_tx"]),
    ("item_cat",    ["cat_tx_cnt", "cat_avg_active_day_gap",
                     "cat_repeat_purchase_rate", "cat_avg_tx"]),
    ("visitor_cat", ["vcat_tx_cnt", "vcat_avg_active_day_gap",
                     "vcat_repeat_purchase_rate", "vcat_avg_tx"]),
    ("vi",          ["vi_tx_cnt"]),
    ("vc",          ["vc_tx_cnt"]),
    ("ic",          ["ic_tx_cnt"]),
    ("ii",          ["ii_co_purchase"]),
])
GPARAM = OrderedDict([
    ("user", "n"), ("item", "m"), ("item_cat", "k"), ("visitor_cat", "x"),
    ("vi", "q"), ("vc", "s"), ("ic", "z"), ("ii", "u"),
])
FEATURES = [f for fs in GROUPS.values() for f in fs]
FEATURE_GROUP = {f: g for g, fs in GROUPS.items() for f in fs}


def domain_grid(k):
    """k значений, равномерно из [7, Y] (целые дни)."""
    vals = np.linspace(WMIN, Y, k)
    out = []
    for v in vals:
        i = int(round(v))
        if i not in out:
            out.append(i)
    return out


def mean_params():
    return {p: MEAN_W for p in GPARAM.values()}


def resolve_windows(params):
    """params может содержать как групповые ('n', ...), так и
    по-признаковые ('f:user_tx_cnt') параметры."""
    out = {}
    for f in FEATURES:
        w = params.get("f:" + f)
        if w is None:
            w = params[GPARAM[FEATURE_GROUP[f]]]
        out[f] = max(WMIN, min(Y, int(w)))
    return out


# ---------------------------------------------------------------- данные
def load_tx(path=DATA_PATH, quick_users=None, seed=0):
    d = pd.read_csv(path, usecols=["itemid", "InvoiceDate", "visitor",
                                   "visitor_cat", "item_cat"],
                    dtype={"itemid": "string", "visitor": "Int64",
                           "visitor_cat": "Int64", "item_cat": "string"})
    d = d[d["visitor"].notna()].copy()
    if quick_users:
        rng = np.random.default_rng(seed)
        us = np.array(sorted(d["visitor"].unique()))
        us = rng.choice(us, size=min(quick_users, len(us)), replace=False)
        d = d[d["visitor"].isin(us)].copy()
    d["visitor"] = d["visitor"].astype(np.int32)
    codes, uniq = pd.factorize(d["itemid"], sort=True)
    d["item"] = codes.astype(np.int32)
    d["vcat"] = d["visitor_cat"].astype(np.int16)
    d["cat"] = d["item_cat"].str.rsplit("_", n=1).str[-1].astype(np.int8)
    d["ts"] = pd.to_datetime(d["InvoiceDate"])
    d["date"] = ((d["ts"] - FIRST_TS).dt.days).astype(np.int16)
    d = d[["visitor", "item", "vcat", "cat", "ts", "date"]]
    d.attrs["items"] = list(uniq)
    return d


def build_candidates(tx, horizon=7, neg_mult=4, seed=42):
    """Кандидаты -- все пары (visitor, item) с историей до t.

    Положительные -- те из них, что куплены в [t, t+horizon); отрицательные --
    случайная подвыборка neg_mult x positives из остальных prior-пар.
    """
    t = T0
    hist = tx[tx["ts"] < t]
    fut = tx[(tx["ts"] >= t) & (tx["ts"] < t + pd.Timedelta(days=horizon))]
    prior = hist[["visitor", "item"]].drop_duplicates()
    bought = fut[["visitor", "item"]].drop_duplicates()
    pos = prior.merge(bought, on=["visitor", "item"], how="inner")
    pos["label"] = 1
    pool = prior.merge(pos[["visitor", "item"]], on=["visitor", "item"],
                       how="left", indicator=True)
    pool = pool[pool["_merge"] == "left_only"][["visitor", "item"]]
    n_neg = int(round(len(pos) * neg_mult))
    n_neg = min(n_neg, len(pool))
    neg = pool.sample(n=n_neg, random_state=seed).copy()
    neg["label"] = 0
    ds = pd.concat([pos, neg], ignore_index=True)
    ds = ds.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    meta_i = tx[["item", "cat"]].drop_duplicates("item")
    meta_v = tx[["visitor", "vcat"]].drop_duplicates("visitor")
    ds = ds.merge(meta_i, on="item", how="left").merge(meta_v, on="visitor", how="left")
    ds["cid"] = np.arange(len(ds), dtype=np.int32)
    return ds


def build_all_candidates(tx, t=T0):
    """Все пары (visitor, item), встречавшиеся до t -- для скоринга."""
    hist = tx[tx["ts"] < t]
    cand = hist[["visitor", "item"]].drop_duplicates().reset_index(drop=True)
    meta_i = tx[["item", "cat"]].drop_duplicates("item")
    meta_v = tx[["visitor", "vcat"]].drop_duplicates("visitor")
    cand = cand.merge(meta_i, on="item", how="left").merge(meta_v, on="visitor", how="left")
    cand["cid"] = np.arange(len(cand), dtype=np.int32)
    return cand


def split_dataset(ds, seed=42):
    idx = np.arange(len(ds))
    tr, te = train_test_split(idx, test_size=0.2, random_state=seed,
                              stratify=ds["label"].to_numpy())
    return tr, te


# ---------------------------------------------------------------- признаки
II_ITEM_CAP = 25          # ограничение размера "корзины" пользователя для ii


class FeatureBuilder:
    """Считает признаки по окну W(t, w) и джойнит их к кандидатам."""

    def __init__(self, tx, cand, t=T0):
        self.tx = tx
        self.cand = cand
        self.t = t
        self._gcache = {}          # (group, window) -> dict
        self._sub_cache = {}       # window -> sub
        self.cand_cid = cand["cid"].to_numpy()
        self.v = cand["visitor"].to_numpy()
        self.i = cand["item"].to_numpy()

    # -------------------------------------------------- utils
    def _sub(self, w):
        if w not in self._sub_cache:
            lo = self.t - pd.Timedelta(days=int(w))
            self._sub_cache[w] = self.tx[(self.tx["ts"] >= lo) &
                                         (self.tx["ts"] < self.t)]
        return self._sub_cache[w]

    def _map(self, series, key, cand_key):
        m = pd.Series(series.to_numpy(), index=series.index)
        out = pd.Series(cand_key).map(m)
        return out.fillna(0.0).to_numpy(np.float32)

    # -------------------------------------------------- группы
    def _group(self, g, w):
        key = (g, int(w))
        if key in self._gcache:
            return self._gcache[key]
        sub = self._sub(int(w))
        out = {}
        if g == "user":
            out = self._agg(sub, "visitor", "user", int(w), repeat=False)
            out = {"user_" + k: v for k, v in out.items()}
        elif g == "item":
            out = self._agg(sub, "item", "item", int(w), repeat=True)
            out = {"item_" + k: v for k, v in out.items()}
        elif g == "item_cat":
            out = self._agg(sub, "cat", "cat", int(w), repeat=True)
            out = {"cat_" + k: v for k, v in out.items()}
        elif g == "visitor_cat":
            out = self._agg(sub, "vcat", "vcat", int(w), repeat=True)
            out = {"vcat_" + k: v for k, v in out.items()}
        elif g == "vi":
            cnt = sub.groupby(["visitor", "item"]).size().rename("c").reset_index()
            out = {"vi_tx_cnt": cnt}
        elif g == "vc":
            cnt = sub.groupby(["visitor", "cat"]).size().rename("c").reset_index()
            out = {"vc_tx_cnt": cnt}
        elif g == "ic":
            cnt = sub.groupby(["item", "vcat"]).size().rename("c").reset_index()
            out = {"ic_tx_cnt": cnt}
        elif g == "ii":
            out = {"ii_co_purchase": self._ii_flag(sub)}
        self._gcache[key] = out
        return out

    def _agg(self, sub, key, prefix, w, repeat):
        g = sub.groupby(key, observed=True)
        tx_cnt = g.size().astype(np.float64)
        days = sub[[key, "date"]].drop_duplicates().sort_values([key, "date"])
        gap = days.groupby(key, observed=True)["date"].diff()
        avg_gap = gap.groupby(days[key], observed=True).mean()
        res = {"tx_cnt": tx_cnt, "avg_active_day_gap": avg_gap,
               "avg_tx": tx_cnt / max(w / 7.0, 1e-9)}
        if repeat:
            vd = sub[[key, "visitor", "date"]].drop_duplicates()
            rep = vd.groupby([key, "visitor"], observed=True).size()
            rep = (rep >= 2).groupby(level=0, observed=True).sum()
            res["repeat_purchase_rate"] = rep.astype(np.float64)
        return res

    def _ii_flag(self, sub):
        b = sub[["visitor", "item"]].drop_duplicates()
        sizes = b.groupby("visitor").size()
        big = sizes[sizes > II_ITEM_CAP].index
        if len(big):
            # оставляем самые частые II_ITEM_CAP товаров у крупных покупателей
            freq = sub[sub["visitor"].isin(big)][["visitor", "item"]].value_counts()
            keep = []
            for vis, grp in freq.groupby(level=0):
                keep.extend([(vis, it) for it in grp.index.get_level_values(1)[:II_ITEM_CAP]])
            keep = pd.MultiIndex.from_tuples(keep, names=["visitor", "item"])
            small = b[~b["visitor"].isin(big)]
            bigb = pd.DataFrame(list(keep), columns=["visitor", "item"])
            b = pd.concat([small, bigb], ignore_index=True)
        b2 = b.merge(b, on="visitor")
        b2 = b2[b2["item_x"] < b2["item_y"]]
        co = b2.groupby(["item_x", "item_y"]).size()
        co = co[co >= 2].reset_index()[["item_x", "item_y"]]
        if len(co) == 0:
            return np.zeros(len(self.cand), dtype=np.float32)
        cb = self.cand[["cid", "visitor", "item"]].merge(
            b.rename(columns={"item": "j"}), on="visitor", how="inner")
        if len(cb) == 0:
            return np.zeros(len(self.cand), dtype=np.float32)
        cb["a"] = np.minimum(cb["item"], cb["j"])
        cb["b"] = np.maximum(cb["item"], cb["j"])
        cb = cb[["cid", "a", "b"]].drop_duplicates()
        hit = cb.merge(co, left_on=["a", "b"], right_on=["item_x", "item_y"],
                       how="inner")
        out = np.zeros(len(self.cand), dtype=np.float32)
        if len(hit):
            out[hit["cid"].unique()] = 1.0
        return out

    # -------------------------------------------------- матрица
    def get_feature(self, f, w):
        g = FEATURE_GROUP[f]
        data = self._group(g, w)[f]
        if isinstance(data, np.ndarray):
            return data
        if f in ("vi_tx_cnt", "vc_tx_cnt", "ic_tx_cnt"):
            keys = {"vi_tx_cnt": ["visitor", "item"],
                    "vc_tx_cnt": ["visitor", "cat"],
                    "ic_tx_cnt": ["item", "vcat"]}[f]
            m = self.cand[keys].merge(data, on=keys, how="left")["c"]
            return m.fillna(0.0).to_numpy(np.float32)
        # Series, индексированная ключом группы
        key = {"user": "visitor", "item": "item", "item_cat": "cat",
               "visitor_cat": "vcat"}[FEATURE_GROUP[f]]
        return self._map(data, key, self.cand[key].to_numpy())

    def matrix(self, params, features=None):
        features = list(features) if features is not None else FEATURES
        windows = resolve_windows(params)
        X = np.zeros((len(self.cand), len(features)), dtype=np.float32)
        for j, f in enumerate(features):
            X[:, j] = self.get_feature(f, windows[f])
        return X, features

    def clear(self, keep_windows=None):
        self._gcache.clear()
        self._sub_cache.clear()


# ---------------------------------------------------------------- модель
LGB_PARAMS = dict(
    n_estimators=120, learning_rate=0.08, num_leaves=31,
    min_child_samples=20, subsample=0.9, subsample_freq=1,
    colsample_bytree=0.9, reg_lambda=1.0, random_state=42,
    n_jobs=8, verbose=-1, deterministic=True, force_row_wise=True,
)


class Evaluator:
    """Обучает модель и логирует все прогоны."""

    def __init__(self, fb, tr, te, tag, models_log=None):
        self.fb = fb
        self.tr = tr
        self.te = te
        self.tag = tag
        self.y = fb.cand["label"].to_numpy()
        self.log = [] if models_log is None else models_log
        self.n_fit = 0
        self.best = (-1.0, -1.0, None, None)
        self.horizon = getattr(fb, "horizon", None)
        self._cache = {}

    def score(self, params, features=None, note=""):
        X, feats = self.fb.matrix(params, features)
        from common import resolve_windows
        wins = resolve_windows(params)
        ckey = (tuple(feats), tuple(int(wins[f]) for f in feats))
        if ckey in self._cache:
            return self._cache[ckey]
        Xtr, Xte = X[self.tr], X[self.te]
        ytr, yte = self.y[self.tr], self.y[self.te]
        m = lgb.LGBMClassifier(**LGB_PARAMS)
        m.fit(Xtr, ytr)
        p = m.predict_proba(Xte)[:, 1]
        self.last_model = m
        self.last_features = list(feats)
        roc = float(roc_auc_score(yte, p))
        pr = float(average_precision_score(yte, p))
        self.n_fit += 1
        rec = {"variant": self.tag, "note": note,
               "n_features": len(feats),
               "features": "|".join(feats),
               "params": json.dumps({k: int(v) for k, v in sorted(params.items())},
                                    ensure_ascii=False),
               "roc": round(roc, 6), "pr": round(pr, 6)}
        self.log.append(rec)
        self._cache[ckey] = (roc, pr)
        if roc > self.best[0] + 1e-9:
            self.best = (roc, pr, dict(params), list(feats))
        return roc, pr

    def improved(self, roc, pr, ref):
        """улучшение относительно ref=(roc, pr)"""
        return roc > ref[0] + 5e-4 or (abs(roc - ref[0]) <= 5e-4 and pr > ref[1] + 5e-4)


def worse(base_roc, base_pr, roc, pr):
    """Правило из задания: модель стала хуже?"""
    if base_roc - roc >= 0.01:
        return True
    if abs(base_roc - roc) < 0.01 and base_pr - pr >= 0.005:
        return True
    return False


def save_csv(rows, name, sep=";"):
    p = os.path.join(RESULTS, name)
    pd.DataFrame(rows).to_csv(p, index=False, sep=sep, encoding="utf-8-sig")
    return p


def window_str(f, windows):
    return f"{GPARAM[FEATURE_GROUP[f]]}={windows[f]}"
