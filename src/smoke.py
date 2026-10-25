# -*- coding: utf-8 -*-
import os, sys, time, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
from common import (load_tx, build_candidates, split_dataset, FeatureBuilder,
                    Evaluator, FEATURES, mean_params, Y, MEAN_W, MODELS)

quick = int(os.environ.get("QUICK_USERS", "0"))
t0 = time.time()
tx = load_tx(quick_users=quick or None)
print(f"tx: {len(tx):,} rows, users={tx.visitor.nunique():,}, items={tx.item.nunique():,}, "
      f"range={tx.ts.min()}..{tx.ts.max()}  ({time.time()-t0:.1f}s)")
ds = build_candidates(tx, horizon=7)
print(f"candidates: {len(ds):,}, positives={int(ds.label.sum()):,} "
      f"({ds.label.mean()*100:.2f}%)  ({time.time()-t0:.1f}s)")
tr, te = split_dataset(ds)
fb = FeatureBuilder(tx, ds)
ev = Evaluator(fb, tr, te, "smoke")
params = mean_params()
print("params:", params, "Y:", Y)
for f in FEATURES:
    t1 = time.time()
    X, feats = fb.matrix(params, [f])
    print(f"  {f:28s} nonzeros={int((X[:,0]!=0).sum()):>7,} "
          f"mean={X[:,0].mean():8.3f}  {time.time()-t1:5.2f}s")
t1 = time.time()
X, feats = fb.matrix(params)
print(f"full matrix {X.shape} in {time.time()-t1:.2f}s")
t1 = time.time()
roc, pr = ev.score(params)
print(f"model: ROC={roc:.4f} PR={pr:.4f}  fit {time.time()-t1:.2f}s")
print(f"TOTAL {time.time()-t0:.1f}s")
