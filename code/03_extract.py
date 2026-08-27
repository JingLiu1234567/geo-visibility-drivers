# -*- coding: utf-8 -*-
"""
03_extract.py — Labelling: one row per (answer, brand)
======================================================
Input : data/clean.parquet, data/brands.csv
Output: data/answer_brand.parquet   labels for every answer x every brand
        data/answers.parquet        one row per answer (brands named, first brand, sources, brand domains cited)
Scope : all 13,542 cleaned answers (extraction is separated from filtering; the main analysis later keeps
        window == 'core' & engine != 'CLAUDE' -> 11,561 answers, of which 8,384 have prose)

Per-row labels (three README definitions + two additions)
----------------------------------------------------------
mentioned        README Mention   : brand name found in the prose, word-bounded, after link markup is removed (0/1)
position         README Position  : rank of the brand's first mention among all lexicon brands named in the answer (1,2,3...; NA if not mentioned)
cited            README Citation  : brand domain appears in source_urls or in an in-text markdown link (0/1)
listed_option    addition         : brand name sits inside a bold span, or within the first 60 characters of a list item / table row / heading (0/1)
named_in_prompt  addition         : brand name appears in the prompt text (0/1)
helpers: cited_rank (1-based index of the first matching URL in source_urls), in_panel (brand belongs to this prompt category's competitor panel)

Link handling
-------------
[anchor](url): the url counts as a citation; the anchor text is kept as prose only when it is <= 3 words and not a
domain or a number (e.g. [cloro](...)); longer anchors (page titles) are removed with the link.
Bare URLs (https://...) are removed from the prose so that domains are not read as brand names.
Sensitivity: the strictest rule (delete every link, anchor included) moves cloro's visibility by <= 0.8 points per engine;
the loosest rule (keep every anchor) inflates Gemini to 16% by counting page titles such as "cloro.dev Review" as mentions.
"""
import os
import re
from urllib.parse import urlparse
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "data")
df = pd.read_parquet(os.path.join(D, "clean.parquet"))
lex = pd.read_csv(os.path.join(D, "brands.csv"))

core = df.reset_index(drop=True)   # all rows

# ---------- compile the lexicon ----------
W1, W2 = r"(?<![A-Za-z0-9])", r"(?![A-Za-z0-9])"
KEYS = list(lex.key)
PATS = {k: re.compile(W1 + "(?:" + rx + ")" + W2, re.I) for k, rx in zip(lex.key, lex.name_regex)}
DOMS = {k: d.split(";") for k, d in zip(lex.key, lex.domains)}
PANEL = {k: (set(p.split(";")) if p != "other" else set()) for k, p in zip(lex.key, lex.panels)}

LINK = re.compile(r"\[([^\]]*)\]\((https?://[^)\s]+)\)")
URL = re.compile(r"https?://\S+")
BOLD = re.compile(r"\*\*(.+?)\*\*", re.S)
LEAD = re.compile(r"^(?:[-*•]|\d+[.)]|#+|\|)\s*\**")


def keep_anchor(t):
    t = t.strip()
    if re.fullmatch(r"\d+", t):
        return False
    if len(t.split()) <= 2 and re.search(r"\.(com|io|dev|ai|org|net|co|so|app)\b", t, re.I):
        return False
    return len(t.split()) <= 3


def to_prose(md):
    p = LINK.sub(lambda m: m.group(1) if keep_anchor(m.group(1)) else " ", md)
    return URL.sub(" ", p)


def host(u):
    try:
        h = urlparse(u).netloc.lower().split("@")[-1].split(":")[0]
    except Exception:
        return ""
    return h[4:] if h.startswith("www.") else h


def dom_hit(h, doms):
    return any(h == d or h.endswith("." + d) for d in doms)


def is_listed(prose, pos, bold_spans, line_starts):
    if any(a <= pos < b for a, b in bold_spans):
        return True
    ls = line_starts[np.searchsorted(line_starts, pos, side="right") - 1]
    nl = prose.find("\n", ls)
    line = prose[ls: nl if nl != -1 else len(prose)]
    m = LEAD.match(line.lstrip())
    if not m:
        return False
    indent = len(line) - len(line.lstrip())
    off = pos - ls - indent - m.end()
    return 0 <= off <= 60


# ---------- brands named in each prompt ----------
named = {}
for pid, txt in df.drop_duplicates("prompt_id")[["prompt_id", "prompt_text"]].itertuples(index=False):
    named[pid] = {k for k, p in PATS.items() if p.search(txt)}

# ---------- label every answer ----------
rows, ans = [], []
for r in core.itertuples(index=False):
    md = r.markdown if isinstance(r.markdown, str) else None
    src_hosts = [host(u) for u in str(r.source_urls).split()] if isinstance(r.source_urls, str) else []
    link_hosts = [host(u) for _, u in LINK.findall(md)] if md else []

    firsts, prose = {}, None
    if md:
        prose = to_prose(md)
        bold_spans = [(m.start(), m.end()) for m in BOLD.finditer(prose)]
        line_starts = np.array([0] + [m.end() for m in re.finditer(r"\n", prose)])
        for k, p in PATS.items():
            m = p.search(prose)
            if m:
                firsts[k] = m.start()
        order = sorted(firsts, key=firsts.get)
        rank = {k: i + 1 for i, k in enumerate(order)}

    n_cited = 0
    for k in KEYS:
        cited_rank = next((i + 1 for i, h in enumerate(src_hosts) if dom_hit(h, DOMS[k])), None)
        cited = int(cited_rank is not None or any(dom_hit(h, DOMS[k]) for h in link_hosts))
        n_cited += cited
        if md:
            mentioned = int(k in firsts)
            position = rank.get(k)
            listed = int(mentioned and is_listed(prose, firsts[k], bold_spans, line_starts))
        else:
            mentioned = position = listed = None
        rows.append((r.result_id, k, mentioned, position, listed, cited, cited_rank,
                     int(k in named[r.prompt_id]), int(r.motion in PANEL[k])))
    ans.append((r.result_id, r.prompt_id, r.engine, r.motion, r.intentType, r.topic, r.day, r.window,
                md is not None, len(firsts) if md else None, order[0] if md and order else None,
                len(src_hosts), n_cited, len(prose) if md else None))

ab = pd.DataFrame(rows, columns=["result_id", "brand", "mentioned", "position", "listed_option",
                                 "cited", "cited_rank", "named_in_prompt", "in_panel"])
for c in ["mentioned", "position", "listed_option", "cited_rank"]:
    ab[c] = ab[c].astype("Int16")
ab.to_parquet(os.path.join(D, "answer_brand.parquet"), index=False)

an = pd.DataFrame(ans, columns=["result_id", "prompt_id", "engine", "motion", "intentType", "topic", "day", "window",
                                "has_prose", "n_brands_named", "first_brand", "n_sources", "n_brands_cited", "prose_len"])
an.to_parquet(os.path.join(D, "answers.parquet"), index=False)

# ---------- QA ----------
pd.set_option("display.width", 200)
print(f"answer_brand: {len(ab):,} rows = {core.shape[0]:,} answers x {len(KEYS)} brands | answers: {len(an):,} rows\n")
P = an[an.has_prose & (an.window == "core") & (an.engine != "CLAUDE")]
print(f"core answers with prose: {len(P):,}; mean brands named {P.n_brands_named.mean():.1f}; zero-brand share {(P.n_brands_named == 0).mean():.1%}")
print("zero-brand share by engine:", (P.groupby("engine").n_brands_named.apply(lambda s: (s == 0).mean())).round(3).to_dict(), "\n")

m = ab.merge(an[["result_id", "engine", "motion", "has_prose", "window"]], on="result_id")
m = m[(m.window == "core") & (m.engine != "CLAUDE")]
c = m[(m.brand == "cloro") & m.has_prose]
print("cloro visibility by engine:", c.groupby("engine").mentioned.mean().round(3).to_dict())
print("cloro cited rate by engine (incl. engines without prose):", m[m.brand == "cloro"].groupby("engine").cited.mean().round(3).to_dict(), "\n")

# spot check: 5 answers, extracted order vs the first 500 characters of prose
samp = P[P.n_brands_named >= 3].sample(5, random_state=7)
for r in samp.itertuples():
    row = core[core.result_id == r.result_id].iloc[0]
    ex = ab[(ab.result_id == r.result_id) & (ab.mentioned == 1)].sort_values("position")
    print("=" * 100)
    print(f"[{r.engine} | {r.motion} | {r.day}] {row.prompt_text}")
    print("extracted:", " -> ".join(f"{b}{'*' if l else ''}{'^' if n else ''}{'(c)' if c_ else ''}"
                                   for b, l, n, c_ in zip(ex.brand, ex.listed_option, ex.named_in_prompt, ex.cited)))
    print("           (* = listed option, ^ = named in prompt, (c) = cited)")
    print("prose:", to_prose(row.markdown)[:500].replace("\n", " | "))
