# cloro GEO take-home — analysis pipeline

What drives (i) the number-one position and (ii) visibility for a brand in generative-engine answers,
using 13 days of answers from 8 AI search engines to 182 prompts in the categories cloro sells into.

The written report (Google Doc) presents the findings; this repository holds the code that produces every
number in it. `METHODS.md` explains the mathematical mechanisms behind each conclusion and maps each
conclusion to the script and table that produced it.

## Pipeline

| step | script | what it does | output |
|---|---|---|---|
| 1 | `code/01_preprocess.py` | parses timestamps and tags, labels time windows, drops rows with neither text nor sources (362), slims columns; prints an audit funnel | `data/clean.parquet`, `data/prompts.csv` |
| 2 | `code/02_brand_lexicon.py` | recognition rules for 140 brands (41 README competitors + cloro + 98 context brands): name regex with word boundaries, case rules and category-term exclusions; domain list for citations; full-corpus match QA | `data/brands.csv` |
| 3 | `code/03_extract.py` | labels every (answer, brand) pair: `mentioned`, `position`, `cited` (README definitions) + `listed_option`, `named_in_prompt` | `data/answer_brand.parquet`, `data/answers.parquet` |
| 4 | `code/04_describe.py` | rankings per motion × engine, cloro's position, day-to-day stability | `out/leaderboard*.csv`, `out/cloro_position.csv`, `out/stability_cloro.csv` |
| 5 | `code/05_drivers.py` | drivers: conditional-probability contrasts and fixed-effects regressions (logit + linear probability model, SE clustered by prompt × engine) | `out/drv_*.csv`, `out/regression.txt` |

## Reproduce

```bash
pip install -r requirements.txt
# put the take-home file at data/results.csv (or set RESULTS_CSV=/path/to/results.csv)
python code/01_preprocess.py
python code/02_brand_lexicon.py
python code/03_extract.py        # ~1 min
python code/04_describe.py
python code/05_drivers.py        # ~2 min
```

The raw file and its derivatives are not committed (`.gitignore`); `data/brands.csv` and everything in `out/`
are committed so the reported numbers can be checked without re-running.

## Data scope used for the main analysis

- **Core window Aug 11–20**: the same 170 prompts ran on 7 engines every day (verified). Aug 9–10 (trial run,
  partial and shifting prompt set) and Aug 21 (collection truncated) are excluded from descriptive rates.
- **Five prose engines** (ChatGPT, Perplexity, Gemini, Copilot, AI Mode): 8,384 answers with text.
  Google and Google News return only URL lists and enter citation figures only. Claude has two days of data
  and is reported separately.
- **Unit of analysis**: (answer, brand) rows — 8,384 answers × each category's competitor panel = 167,276 rows.

## Key results (core window, five engines pooled)

- Being cited is the strongest driver of a mention: P(mention | cited) = 43.9% vs P(mention | not cited) = 4.8%;
  odds ratio 41 (logit with brand/engine/topic/intent fixed effects), +32 pp (LPM with prompt fixed effects).
  The effect ranges from +15 pp on Perplexity to +51 pp on Copilot.
- The first position is driven by being the answer's **first-ranked source** (71.8% vs 20.9%; OR 11, +40 pp),
  far more than by being cited at all (+6 pp). Prompts that name a brand hand it the first position 94% of the
  time (a "target" effect, excluded from first-position rates).
- cloro is cited at top-tier rates (2nd, 2nd and 4th in its three categories) but converts citations to mentions
  at 15–31% against 54–94% for the leaders, and is almost never mentioned without a citation (≤ 2.2% vs 11–60%).

See `METHODS.md` for definitions, formulas, model specifications and confidence intervals.
