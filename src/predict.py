# -*- coding: utf-8 -*-
"""Применение обученной модели к произвольному snapshot.

Пример:
    py src/predict.py --model models/final_v1_grid5_h7.json \
        --snapshot "2011-11-20 00:00" --out predictions.csv

Скрипт читает конфиг модели (features + windows), считает те же признаки
по тем же окнам W(t, w) = [t - w, t) и оценивает вероятность покупки
для всех пар (visitor, item), встречавшихся до snapshot.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lightgbm as lgb
import numpy as np
import pandas as pd

from common import FeatureBuilder, build_all_candidates, load_tx, mean_params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="json-конфиг модели")
    ap.add_argument("--model-file", default=None, help="txt с бустером "
                    "(по умолчанию рядом с конфигом)")
    ap.add_argument("--snapshot", default=None,
                    help="точка отсчёта, по умолчанию из конфига/T0")
    ap.add_argument("--out", default="predictions.csv")
    ap.add_argument("--top", type=int, default=0,
                    help="оставить top-N рекомендаций на пользователя")
    ap.add_argument("--data", default=None, help="путь к CSV с транзакциями")
    args = ap.parse_args()

    with open(args.model, encoding="utf-8") as fh:
        cfg = json.load(fh)
    model_file = args.model_file or os.path.splitext(args.model)[0] + ".txt"
    with open(model_file, encoding="utf-8") as fh:
        booster = lgb.Booster(model_str=fh.read())

    ts = pd.Timestamp(args.snapshot) if args.snapshot else pd.Timestamp(
        cfg.get("snapshot", "2011-11-20 00:00:00"))
    tx = load_tx(args.data) if args.data else load_tx()
    cand = build_all_candidates(tx, t=ts)
    fb = FeatureBuilder(tx, cand, t=ts)
    windows = {}
    for f in cfg["features"]:
        w = cfg["windows"].get(f, cfg["windows"].get("f:" + f))
        if w is None:
            raise KeyError(f"нет окна для признака {f} в {args.model}")
        windows[f] = int(w)
    params = mean_params()
    params.update({"f:" + f: w for f, w in windows.items()})
    X, feats = fb.matrix(params, cfg["features"])
    cand["score"] = booster.predict(X)
    out = cand[["visitor", "item", "score"]].sort_values(
        ["visitor", "score"], ascending=[True, False])
    if args.top:
        out = out.groupby("visitor", observed=True).head(args.top)
    out.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"snapshot={ts}, пар={len(out):,}, файл={args.out}")


if __name__ == "__main__":
    main()
