# -*- coding: utf-8 -*-
"""
05_drivers.py — What drives visibility and the first position
=============================================================
Question i : first position   -> Y = (position == 1)
Question ii: visibility       -> Y = mentioned
Unit  : (answer, brand) rows; window == 'core' & engine != 'CLAUDE' & prose present (8,384 answers)
        x the competitor panel of the prompt's category -> 167,276 rows
Candidate drivers: cited / cited_rank / named_in_prompt / intentType / topic / engine /
                   answer length / number of sources / number of brands named / brand strength
Method: A-E  contrasts of conditional probabilities, one factor at a time
        F    fixed-effects regressions (brand, engine, topic/intent or prompt), standard errors clustered by prompt x engine
Output: out/drv_*.csv, out/regression.txt
"""
import os
import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D, O = os.path.join(ROOT, "data"), os.path.join(ROOT, "out")
ab = pd.read_parquet(os.path.join(D, "answer_brand.parquet"))
an = pd.read_parquet(os.path.join(D, "answers.parquet"))
lex = pd.read_csv(os.path.join(D, "brands.csv")).set_index("key")
pd.set_option("display.width", 220); pd.set_option("display.max_rows", 300)
pct = lambda v: f"{v * 100:.1f}%"

M = an[(an.window == "core") & (an.engine != "CLAUDE") & an.has_prose]
x = ab.merge(M[["result_id", "prompt_id", "engine", "motion", "intentType", "topic", "n_brands_named", "n_sources", "prose_len"]], on="result_id")
x = x[x.in_panel == 1].copy()
x["mentioned"] = x.mentioned.astype(int)
x["top1"] = x.position.eq(1).fillna(False).astype(int)
x["cited"] = x.cited.astype(int)
x["name"] = x.brand.map(lex.name)
print(f"analysis rows: {len(x):,} (answer x panel brand) from {x.result_id.nunique():,} answers\n")

# ======================= A. citation -> mention =======================
print("=" * 95); print("A. Does being cited drive being mentioned?  P(mention | cited) vs P(mention | not cited)"); print("=" * 95)
a = x.groupby("cited").mentioned.mean()
print(f"all brands: not cited {pct(a[0])}  ->  cited {pct(a[1])}   (lift {a[1] / a[0]:.1f}x)\n")
t = x.groupby(["engine", "cited"]).mentioned.mean().unstack()
t.columns = ["P(mention|not cited)", "P(mention|cited)"]
t["lift"] = (t.iloc[:, 1] / t.iloc[:, 0]).round(1)
t["share cited"] = x.groupby("engine").cited.mean()
print("by engine:")
print(t.assign(**{c: t[c].map(pct) for c in t.columns[:2]}, **{"share cited": t["share cited"].map(pct)}).to_string(), "\n")
t2 = x.groupby(["motion", "cited"]).mentioned.mean().unstack()
t2.columns = ["P(mention|not cited)", "P(mention|cited)"]
t2["lift"] = (t2.iloc[:, 1] / t2.iloc[:, 0]).round(1)
print("by motion:")
print(t2.assign(**{c: t2[c].map(pct) for c in t2.columns[:2]}).to_string(), "\n")
t.to_csv(os.path.join(O, "drv_citation_by_engine.csv"), encoding="utf-8-sig")

xr = x[x.cited == 1].copy()
xr["rank_bin"] = pd.cut(xr.cited_rank.astype(float), [0, 1, 3, 10, 999], labels=["source #1", "source #2-3", "source #4-10", "source #11+"]).astype(str)
xr.loc[xr.cited_rank.isna(), "rank_bin"] = "in-text link only"
print("position of the citation -> P(mention), cited rows only:")
print(xr.groupby("rank_bin", observed=True).agg(n=("mentioned", "size"), p_mention=("mentioned", "mean"))
        .assign(p_mention=lambda d: d.p_mention.map(pct)).to_string(), "\n")

# ======================= B. brand level: conversion and brand memory =======================
print("=" * 95); print("B. Brand level: citation-to-mention conversion, and mentions without any citation (brand memory)"); print("=" * 95)
bl = x.groupby(["motion", "brand"]).apply(lambda g: pd.Series({
    "n": len(g), "visibility": g.mentioned.mean(), "cited_rate": g.cited.mean(),
    "P(mention|cited)": g[g.cited == 1].mentioned.mean() if g.cited.sum() else np.nan,
    "P(mention|not cited)": g[g.cited == 0].mentioned.mean(),
    "share of mentions with citation": g[g.mentioned == 1].cited.mean() if g.mentioned.sum() else np.nan}), include_groups=False).reset_index()
bl["name"] = bl.brand.map(lex.name)
bl.round(4).to_csv(os.path.join(O, "drv_brand_conversion.csv"), index=False, encoding="utf-8-sig")
COLS = ["visibility", "cited_rate", "P(mention|cited)", "P(mention|not cited)", "share of mentions with citation"]
for mo, g in bl.groupby("motion"):
    g = g.sort_values("visibility", ascending=False)
    show = pd.concat([g.head(6), g[g.brand == "cloro"]]).drop_duplicates("brand")
    print(f"[{mo}]")
    print(show[["name"] + COLS].assign(**{c: show[c].map(pct) for c in COLS}).to_string(index=False), "\n")

# ======================= C. prompt features =======================
print("=" * 95); print("C. Prompt features: intent / topic / brand named in the prompt"); print("=" * 95)
ans_any = x.groupby(["result_id", "intentType", "motion", "topic", "engine"]).mentioned.max().reset_index()
print("share of answers that name at least one panel brand (does this kind of prompt elicit brand names?):")
print(ans_any.groupby(["motion", "intentType"]).mentioned.mean().unstack().map(pct).to_string(), "\n")
tp = ans_any.groupby(["motion", "topic"]).mentioned.agg(["size", "mean"]).reset_index().sort_values(["motion", "mean"], ascending=[True, False])
tp["mean"] = tp["mean"].map(pct)
print("by topic (share of answers naming any panel brand; size = answers):")
for mo, g in tp.groupby("motion"):
    print(f"  [{mo}] high: " + ", ".join(f"{r.topic} {r['mean']}" for _, r in g.head(4).iterrows()))
    print(f"  [{mo}] low : " + ", ".join(f"{r.topic} {r['mean']}" for _, r in g.tail(4).iterrows()))
tp.to_csv(os.path.join(O, "drv_topic.csv"), index=False, encoding="utf-8-sig")
print()
nm = x.groupby("named_in_prompt").agg(n=("mentioned", "size"), p_mention=("mentioned", "mean"), p_first=("top1", "mean"))
print("brand named in the prompt vs not:")
print(nm.assign(p_mention=nm.p_mention.map(pct), p_first=nm.p_first.map(pct)).to_string(), "\n")
print("cloro by intent:", x[x.brand == "cloro"].groupby("intentType").mentioned.mean().map(pct).to_dict())
print("prompts that name cloro:", x[(x.brand == "cloro") & (x.named_in_prompt == 1)].prompt_id.nunique(), "\n")

# ======================= D. answer features =======================
print("=" * 95); print("D. Answer features: number of brands named / answer length (cloro vs all panel brands)"); print("=" * 95)
x["nb_bin"] = pd.cut(x.n_brands_named, [-1, 0, 2, 5, 99], labels=["0", "1-2", "3-5", "6+"])
x["len_bin"] = pd.qcut(x.prose_len, 4, labels=["Q1 short", "Q2", "Q3", "Q4 long"])
for col, lab in [("nb_bin", "brands named in the answer"), ("len_bin", "answer length quartile")]:
    g = x.groupby(col, observed=True).agg(p_mention_all=("mentioned", "mean"), p_first_all=("top1", "mean"))
    g["p_mention_cloro"] = x[x.brand == "cloro"].groupby(col, observed=True).mentioned.mean()
    print(f"{lab}:"); print(g.map(pct).to_string(), "\n")

# ======================= E. first position, given mentioned =======================
print("=" * 95); print("E. Question i: among mentioned brands, which get the first position?"); print("=" * 95)
m1 = x[x.mentioned == 1].copy()
print(f"mentioned rows: {len(m1):,}; first position: {pct(m1.top1.mean())}\n")
g = m1.groupby("cited").top1.mean(); print(f"cited: {pct(g[1])} vs not cited: {pct(g[0])}")
g = m1.groupby("named_in_prompt").top1.mean(); print(f"named in prompt: {pct(g[1])} vs not: {pct(g[0])}")
g = m1.groupby("listed_option").top1.mean(); print(f"listed as option: {pct(g[1])} vs mentioned in passing: {pct(g[0])}")
m1["first_cited"] = (m1.cited_rank == 1).fillna(False).astype(int)
g = m1.groupby("first_cited").top1.mean(); print(f"is the first-ranked source: {pct(g[1])} vs not: {pct(g[0])}")
g = m1.groupby("nb_bin", observed=True).top1.mean(); print("by brands named in the answer:", g.map(pct).to_dict(), "\n")
vis = x.groupby(["motion", "engine", "brand"]).mentioned.mean().rename("brand_vis").reset_index()
m1 = m1.merge(vis, on=["motion", "engine", "brand"])
m1["vis_bin"] = pd.cut(m1.brand_vis, [0, .05, .15, .4, 1], labels=["<5%", "5-15%", "15-40%", ">40%"])
print("by brand strength (the brand's visibility in that motion x engine cell):", m1.groupby("vis_bin", observed=True).top1.mean().map(pct).to_dict(), "\n")

# ======================= F. regressions =======================
print("=" * 95); print("F. Regressions: brand / engine / prompt controls, SE clustered by prompt x engine"); print("=" * 95)
print("  Prompt fixed effects cause perfect separation in the logit (some prompts never name any brand),")
print("  so the logit uses topic + intent fixed effects and a linear probability model (OLS) keeps prompt fixed effects")
print("  as the cross-check. Coefficients: odds ratios (logit) and percentage points (LPM).\n")
x["log_len"] = np.log1p(x.prose_len); x["log_src"] = np.log1p(x.n_sources)
x["cl"] = x.prompt_id + "_" + x.engine
lines = []
HIDE = ("C(brand)", "C(prompt_id)", "C(engine)", "C(topic)", "C(motion)", "Intercept")


def run_logit(formula, data):
    try:
        return smf.logit(formula, data=data).fit(disp=0, maxiter=300, cov_type="cluster", cov_kwds={"groups": data.cl})
    except Exception:
        return None


def run_ols(formula, data):
    return smf.ols(formula, data=data).fit(cov_type="cluster", cov_kwds={"groups": data.cl})


def show(mod, label, kind):
    keep = [i for i in mod.params.index if not i.startswith(HIDE)]
    if kind == "logit":
        out = pd.DataFrame({"coef": mod.params[keep], "odds_ratio": np.exp(mod.params[keep]), "p": mod.pvalues[keep]}).round(3)
        s = f"\n[{label}]  n={int(mod.nobs):,}  pseudo-R2={mod.prsquared:.3f}\n{out.to_string()}"
    else:
        out = pd.DataFrame({"coef (pct points)": (mod.params[keep] * 100).round(2), "p": mod.pvalues[keep].round(3)})
        s = f"\n[{label}]  n={int(mod.nobs):,}  R2={mod.rsquared:.3f}\n{out.to_string()}"
    print(s); lines.append(s)


ok_brand = x.groupby("brand").mentioned.sum()
xs = x[x.brand.isin(ok_brand[ok_brand > 0].index)].copy()   # brands never mentioned carry no information for the logit

# ---- question ii: visibility ----
m = run_logit("mentioned ~ cited + named_in_prompt + log_len + log_src + n_brands_named + C(intentType) + C(brand) + C(engine) + C(topic)", xs)
if m is not None:
    show(m, "question ii  Y=mentioned  logit (brand / engine / topic / intent FE)", "logit")
show(run_ols("mentioned ~ cited + named_in_prompt + log_len + log_src + n_brands_named + C(brand) + C(engine) + C(prompt_id)", xs),
     "question ii  Y=mentioned  OLS-LPM (brand / engine / prompt FE)", "ols")
rows = []
for e, g in xs.groupby("engine"):
    gb = g.groupby("brand").mentioned.sum(); g = g[g.brand.isin(gb[gb > 0].index)]
    m = run_logit("mentioned ~ cited + named_in_prompt + log_len + n_brands_named + C(intentType) + C(motion) + C(brand)", g)
    orr = round(float(np.exp(m.params["cited"])), 1) if m is not None else "separated"
    lpm = run_ols("mentioned ~ cited + named_in_prompt + log_len + n_brands_named + C(brand) + C(prompt_id)", g)
    rows.append((e, orr, round(float(lpm.params["cited"]) * 100, 1), round(float(lpm.pvalues["cited"]), 4)))
s = ("\n[question ii  effect of `cited` by engine]  OR = logit odds ratio (brand / motion / intent FE); "
     "LPM = percentage-point increase in P(mention) (prompt FE)\n"
     + pd.DataFrame(rows, columns=["engine", "OR_cited", "LPM_pct_points", "p"]).to_string(index=False))
print(s); lines.append(s)

# ---- question i: first position, given mentioned ----
m1["first_cited"] = m1.first_cited.astype(int); m1["log_len"] = np.log1p(m1.prose_len); m1["cl"] = m1.prompt_id + "_" + m1.engine
gb = m1.groupby("brand").top1.sum(); m1s = m1[m1.brand.isin(gb[gb > 0].index)].copy()
m = run_logit("top1 ~ cited + first_cited + named_in_prompt + listed_option + n_brands_named + C(intentType) + C(brand) + C(engine) + C(topic)", m1s)
if m is not None:
    show(m, "question i   Y=top1 | mentioned  logit", "logit")
else:
    print("  (question i logit separated; see LPM)")
show(run_ols("top1 ~ cited + first_cited + named_in_prompt + listed_option + n_brands_named + C(brand) + C(engine) + C(prompt_id)", m1s),
     "question i   Y=top1 | mentioned  OLS-LPM (prompt FE)", "ols")

# ---- cloro only ----
c = x[x.brand == "cloro"].copy()
m = run_logit("mentioned ~ cited + log_len + n_brands_named + C(engine) + C(motion) + C(intentType)", c)
if m is not None:
    s = f"\n[cloro only  Y=mentioned  n={int(m.nobs):,}]\n" + pd.DataFrame({"odds_ratio": np.exp(m.params), "p": m.pvalues}).round(3).to_string()
    print(s); lines.append(s)
with open(os.path.join(O, "regression.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("\nregression output written to out/regression.txt")
