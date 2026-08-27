# -*- coding: utf-8 -*-
"""
04_describe.py — Descriptive results: rankings / cloro's position / stability
=============================================================================
Input : data/answer_brand.parquet, data/answers.parquet, data/brands.csv
Population: window == 'core' & engine != 'CLAUDE' & prose present -> 8,384 answers;
            brands = the competitor panel of the prompt's category (motion) plus cloro
Output: out/leaderboard.csv         metrics per motion x engine x brand
        out/leaderboard_pooled.csv  metrics per motion x brand (five engines pooled)
        out/cloro_position.csv      cloro's rank and gap to the leader in each cell
        out/stability_cloro.csv     for each (prompt, engine): days observed and days cloro was mentioned

Metric definitions (all are means of a binary label over the answers in the group)
----------------------------------------------------------------------------------
visibility   = P(mentioned = 1)                                     (README Visibility)
top1_rate    = P(position = 1)                                      (README Position = 1)
top1_clean   = P(position = 1 and named_in_prompt = 0)              (removes the "target" effect of prompts that name a brand)
avg_pos      = mean position when mentioned
cited_rate   = P(cited = 1)                                         (README Citation)
listed_rate  = P(listed_option = 1)
"""
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D, O = os.path.join(ROOT, "data"), os.path.join(ROOT, "out")
os.makedirs(O, exist_ok=True)
ab = pd.read_parquet(os.path.join(D, "answer_brand.parquet"))
an = pd.read_parquet(os.path.join(D, "answers.parquet"))
lex = pd.read_csv(os.path.join(D, "brands.csv")).set_index("key")

M = an[(an.window == "core") & (an.engine != "CLAUDE") & an.has_prose]
x = ab.merge(M[["result_id", "prompt_id", "engine", "motion", "day"]], on="result_id")
x = x[x.in_panel == 1].copy()
x["top1"] = x.position.eq(1).fillna(False).astype(int)
x["top1_clean"] = (x.position.eq(1).fillna(False) & (x.named_in_prompt == 0)).astype(int)


def agg(g):
    return pd.Series({
        "n": len(g), "visibility": g.mentioned.mean(), "top1_rate": g.top1.mean(),
        "top1_clean": g.top1_clean.mean(), "avg_pos": g.position.mean(),
        "cited_rate": g.cited.mean(), "listed_rate": g.listed_option.mean(),
        "named_share": g.named_in_prompt.mean()})


lb = x.groupby(["motion", "engine", "brand"]).apply(agg, include_groups=False).reset_index()
lb["name"] = lb.brand.map(lex.name)
lb["rank_vis"] = lb.groupby(["motion", "engine"]).visibility.rank(ascending=False, method="min").astype(int)
lb["rank_top1"] = lb.groupby(["motion", "engine"]).top1_clean.rank(ascending=False, method="min").astype(int)
lb.round(4).to_csv(os.path.join(O, "leaderboard.csv"), index=False, encoding="utf-8-sig")

pooled = x.groupby(["motion", "brand"]).apply(agg, include_groups=False).reset_index()
pooled["name"] = pooled.brand.map(lex.name)
pooled["rank_vis"] = pooled.groupby("motion").visibility.rank(ascending=False, method="min").astype(int)
pooled.round(4).to_csv(os.path.join(O, "leaderboard_pooled.csv"), index=False, encoding="utf-8-sig")

# ---- cloro's position ----
cp = lb[lb.brand == "cloro"][["motion", "engine", "n", "visibility", "rank_vis", "top1_clean", "rank_top1", "cited_rate"]].copy()
lead = lb.sort_values("visibility", ascending=False).groupby(["motion", "engine"]).head(1)[["motion", "engine", "name", "visibility"]]
lead.columns = ["motion", "engine", "leader", "leader_vis"]
cp = cp.merge(lead, on=["motion", "engine"])
cp["n_panel"] = cp.motion.map(lb.groupby("motion").brand.nunique())
cp.round(4).to_csv(os.path.join(O, "cloro_position.csv"), index=False, encoding="utf-8-sig")

# ---- stability: days cloro is mentioned per (prompt, engine) over the 10 core days ----
st = (x[x.brand == "cloro"].groupby(["motion", "engine", "prompt_id"])
      .agg(days=("day", "nunique"), days_mentioned=("mentioned", "sum")).reset_index())
st["share"] = st.days_mentioned / st.days
st.to_csv(os.path.join(O, "stability_cloro.csv"), index=False, encoding="utf-8-sig")

# ================= print =================
pd.set_option("display.width", 220); pd.set_option("display.max_rows", 200)
pct = lambda v: f"{v * 100:.1f}%"

print("=" * 90); print("A. Rankings per motion (five engines pooled, by visibility; top 10 + cloro)"); print("=" * 90)
for mo, g in pooled.groupby("motion"):
    g = g.sort_values("visibility", ascending=False)
    show = pd.concat([g.head(10), g[g.brand == "cloro"]]).drop_duplicates("brand")
    print(f"\n[{mo}]  panel of {len(g)} brands, n = {int(g.n.iloc[0])} answers per brand")
    t = show[["rank_vis", "name", "visibility", "top1_rate", "top1_clean", "avg_pos", "cited_rate", "listed_rate"]].copy()
    for c in ["visibility", "top1_rate", "top1_clean", "cited_rate", "listed_rate"]:
        t[c] = t[c].map(pct)
    t["avg_pos"] = t.avg_pos.round(1)
    print(t.to_string(index=False))

print("\n" + "=" * 90); print("B. cloro in each of the 15 motion x engine cells"); print("=" * 90)
t = cp.copy()
for c in ["visibility", "top1_clean", "cited_rate", "leader_vis"]:
    t[c] = t[c].map(pct)
t["rank_vis"] = t.rank_vis.astype(str) + "/" + t.n_panel.astype(str)
print(t[["motion", "engine", "n", "visibility", "rank_vis", "top1_clean", "rank_top1", "cited_rate", "leader", "leader_vis"]].to_string(index=False))

print("\n" + "=" * 90); print("C. Brand with the highest first-position rate per cell (top1_clean)"); print("=" * 90)
top = lb.sort_values("top1_clean", ascending=False).groupby(["motion", "engine"]).head(1)
print(top.pivot(index="motion", columns="engine", values="name").to_string())
print()
print(top.pivot(index="motion", columns="engine", values="top1_clean").map(pct).to_string())

print("\n" + "=" * 90); print("D. cloro stability: share of the 10 days on which each (prompt, engine) pair mentions cloro"); print("=" * 90)
bins = pd.cut(st.share, [-0.01, 0, 0.3, 0.7, 1.0], labels=["never", "occasional (<=30%)", "half (30-70%)", "stable (>70%)"])
print(pd.crosstab(st.engine, bins).to_string())
print(f"\n{len(st)} (prompt, engine) pairs; mentioned on >= 1 day: {(st.share > 0).sum()}; stable (> 70% of days): {(st.share > 0.7).sum()}")
print(f"cloro visibility {x[x.brand == 'cloro'].mentioned.mean() * 100:.1f}% = coverage {(st.share > 0).mean() * 100:.1f}% "
      f"(pairs with >= 1 mention) x conditional stability {st[st.share > 0].share.mean() * 100:.1f}% (mean share of days within those pairs)")
