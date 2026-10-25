# -*- coding: utf-8 -*-
"""Подготовка отредактированной retail-таблицы из исходного CSV.

Правила (из задания к таблице):
 1. оставить строки, где StockCode начинается с цифр; StockCode -> itemid;
 2. Customer ID -> visitor;
 3. удалить Invoice и Description;
 4. Country -> visitor_cat (уникальным строкам -- уникальные числовые id);
 5. item_cat: 10 квантильных интервалов price (5-20% строк в каждом),
    price удаляется.
Вспомогательная таблица для воспроизводимости.
"""
import sys

import numpy as np
import pandas as pd

SRC = sys.argv[1] if len(sys.argv) > 1 else "online_retail_II.csv"
DST = sys.argv[2] if len(sys.argv) > 2 else "data/online_retail_II_edited.csv"

df = pd.read_csv(SRC, dtype={"StockCode": "string"}, encoding="utf-8")
df = df[df["StockCode"].astype("string").fillna("").str.match(r"^\d+")].copy()
df = df.rename(columns={"StockCode": "itemid", "Customer ID": "visitor"})
df = df.drop(columns=["Invoice", "Description"])
countries = sorted(df["Country"].dropna().unique())
df["visitor_cat"] = df["Country"].map({c: i + 1 for i, c in enumerate(countries)})
df = df.drop(columns=["Country"])

price = df["Price"].astype(float)
vals = np.sort(price.unique())
counts = price.value_counts().reindex(vals).to_numpy()
N = len(df)
cum = np.concatenate([[0], np.cumsum(counts)]).astype(float)
m = len(vals)
K = 10
share = (cum[None, :] - cum[:, None]) / N
valid = (np.triu(np.ones((m + 1, m + 1), dtype=bool), k=1)
         & (share >= 0.05) & (share <= 0.20))
cost = np.where(valid, (share - 0.10) ** 2, np.inf)
dp = np.full((K + 1, m + 1), np.inf)
prev = np.full((K + 1, m + 1), -1, dtype=int)
dp[0, 0] = 0.0
for k in range(1, K + 1):
    for j in range(1, m + 1):
        cand = dp[k - 1, :j] + cost[:j, j]
        i = int(np.argmin(cand))
        if np.isfinite(cand[i]):
            dp[k, j] = cand[i]
            prev[k, j] = i
if not np.isfinite(dp[K, m]):
    raise SystemExit("не удалось разбить price на 10 интервалов 5-20%")
cuts = [m]
j = m
for k in range(K, 0, -1):
    j = prev[k, j]
    cuts.append(j)
cuts = cuts[::-1]
labels = np.empty(N, dtype=object)
for k in range(K):
    lo, hi = vals[cuts[k]], vals[cuts[k + 1] - 1]
    labels[((price >= lo) & (price <= hi)).to_numpy()] = f"item_cat_{k + 1}"
df["item_cat"] = labels
df = df.drop(columns=["Price"])
df = df[["itemid", "Quantity", "InvoiceDate", "visitor", "visitor_cat", "item_cat"]]
df.to_csv(DST, index=False, encoding="utf-8-sig")
print(f"saved {DST}: {len(df):,} rows")
