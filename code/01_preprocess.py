# -*- coding: utf-8 -*-
"""
01_preprocess.py — Data preprocessing
=====================================
Input : data/results.csv   raw take-home file (override the path with env var RESULTS_CSV)
Output: data/clean.parquet, data/clean.csv   analysable rows, 20 columns
        data/prompts.csv                      lookup table of the 182 prompts
Run   : python code/01_preprocess.py          (prints an audit funnel)

Rules
-----
P1 time       created_at -> datetime (formats are mixed, parsed as ISO8601); derive `day`
P2 tags       tags -> motion / intentType / source
P3 dtypes     the four count columns -> nullable Int32 (missing stays missing, not zero)
P4 windows    early = Aug 09-10 (trial run, partial and shifting prompt set)
              core  = Aug 11-20 (7 engines x the same 170 prompts x 10 days; verified identical every day)
              tail  = Aug 21    (collection stopped after 71 prompts)
P5 soft fail  status = SUCCESS but the text is an engine error message (2 rows) -> markdown set to NA
P6 drop       rows with neither markdown nor source_urls
              = 156 FAILED (all content fields empty) + 206 SUCCESS rows where the engine returned nothing
P7 slim       drop redundant columns: hour / tags (already split) / in_core_promptset (kept in prompts.csv) / created_at (day is kept)

Kept on purpose
  GOOGLE / GOOGLE_NEWS rows have no prose but do carry source_urls  -> citation analysis only
  zero-brand answers and very short answers are valid content       -> kept (they are the visibility denominator)

Analysis populations (no flag columns; each script applies its own filter)
  mention / position : window == 'core' & engine != 'CLAUDE' & markdown not null    -> 8,384 rows
  citation           : window == 'core' & engine != 'CLAUDE' & source_urls not null -> 11,133 rows
"""
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data")
RAW = os.environ.get("RESULTS_CSV", os.path.join(OUT, "results.csv"))

df = pd.read_csv(RAW)
n0 = len(df)

# P1 time
df["created_at"] = pd.to_datetime(df["created_at"], format="ISO8601")
df["day"] = df["created_at"].dt.date.astype("str")
df["hour"] = df["created_at"].dt.hour

# P2 tags
for g in ["motion", "intentType", "source"]:
    df[g] = df["tags"].str.extract(g + r":([^;]+)")[0]
assert df[["motion", "intentType", "source"]].notna().all().all(), "prompt with missing tag"

# P3 dtypes
for c in ["source_count", "sources_with_description_count", "related_query_count", "markdown_length"]:
    df[c] = df[c].astype("Int32")

# P4 windows
df["window"] = np.select([df.day <= "2026-08-10", df.day <= "2026-08-20"], ["early", "core"], "tail")
df["in_core_promptset"] = df.prompt_id.isin(set(df.loc[df.window == "core", "prompt_id"]))

# audit: failure rates are computed before rows are dropped
core7 = df[(df.window == "core") & (df.engine != "CLAUDE")]
fail_rate = (1 - core7.groupby("engine").status.apply(lambda s: s.eq("SUCCESS").mean())).round(4)

# P5 soft failures: error text returned with status = SUCCESS
SOFT = r"(?i)^I'?m sorry, I'?m having trouble responding|^I can'?t browse right now"
soft = df.markdown.fillna("").str.contains(SOFT, regex=True)
df.loc[soft, "markdown"] = pd.NA
df.loc[soft, "markdown_length"] = pd.NA

# P6 drop rows with neither text nor sources
drop = df.markdown.isna() & df.source_urls.isna()
detail = df[drop].groupby(["engine", "status"]).size()
df = df[~drop].copy()

prompts = (df.sort_values("created_at").drop_duplicates("prompt_id")
             [["prompt_id", "prompt_text", "motion", "intentType", "source", "topic", "in_core_promptset"]])
prompts["days_observed"] = prompts.prompt_id.map(df.groupby("prompt_id").day.nunique())
prompts.to_csv(os.path.join(OUT, "prompts.csv"), index=False)

# P7 slim
df = df.drop(columns=["hour", "tags", "in_core_promptset", "created_at"])
df.to_parquet(os.path.join(OUT, "clean.parquet"), index=False)
df.to_csv(os.path.join(OUT, "clean.csv"), index=False, encoding="utf-8-sig")

# ---- audit funnel ----
print("=" * 60)
print(f"R0 raw rows                                 {n0:>6}")
print(f"R1 soft failures (markdown set to NA)       {int(soft.sum()):>6}")
print(f"R2 dropped: no text and no sources          {int(drop.sum()):>6}, breakdown:")
print(detail.to_string())
print(f"R3 analysable rows kept                     {len(df):>6}")
w = df.window
print(f"R4 windows   early {(w == 'early').sum():>5} | core {(w == 'core').sum():>6} | tail {(w == 'tail').sum():>4}")
m = df[(w == "core") & (df.engine != "CLAUDE")]
print(f"R5 core non-Claude {len(m):>6} -> with prose {m.markdown.notna().sum():>5} (mention/position) | with sources {m.source_urls.notna().sum():>6} (citation)")
print("\nmention/position population, motion x engine:")
mp = m[m.markdown.notna()]
print(pd.crosstab(mp.motion, mp.engine).to_string())
print("\n[archived] engine failure rate in the core window, before dropping:")
print(fail_rate.to_string())
print(f"\nwritten: clean.parquet ({len(df)} rows x {df.shape[1]} cols), clean.csv, prompts.csv ({len(prompts)} prompts)")
