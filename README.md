# GEO visibility drivers — analysis pipeline

What drives (i) the number-one position and (ii) visibility for a brand in generative-engine answers,
using 13 days of answers from 8 AI search engines to 182 prompts across three product categories
(SERP APIs, LLM-answer scraping APIs, AI-visibility tracking APIs). The brand under study is cloro.

The written report (Google Doc) presents the findings; this repository holds the code that produces every
number in it. `METHODS.md` explains how each number was computed and maps each conclusion to the script and
table that produced it.

## Pipeline

| step | script | what it does | output |
|---|---|---|---|
| 1 | `code/01_preprocess.py` | parses timestamps and tags, labels time windows, drops rows with neither text nor sources (362), slims columns; prints an audit funnel | `data/clean.parquet`, `data/prompts.csv` |
| 2 | `code/02_brand_lexicon.py` | recognition rules for 140 brands (41 README competitors + cloro + 98 context brands): name regex with word boundaries, case rules and category-term exclusions; domain list for citations; full-corpus match QA | `data/brands.csv` |
| 3 | `code/03_extract.py` | labels every (answer, brand) pair: `mentioned`, `position`, `cited` (README definitions) + `listed_option`, `named_in_prompt` | `data/answer_brand.parquet`, `data/answers.parquet` |
| 4 | `code/04_describe.py` | rankings per motion × engine, cloro's position, day-to-day stability | `out/leaderboard*.csv`, `out/cloro_position.csv`, `out/stability_cloro.csv` |
| 5 | `code/05_drivers.py` | drivers: conditional rate contrasts (one factor at a time), plus an additional regression check | `out/drv_*.csv`, `out/regression.txt` |

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

- Being cited is the strongest driver of visibility: a brand is mentioned in 43.9% of the answers that cite
  one of its pages, against 4.8% of the answers that do not.
- The first position is driven by citation order: among mentioned brands, the first-position rate is 71.8%
  when the brand is the answer's first-ranked source, against 20.9% otherwise.
- cloro is cited at top-tier rates (2nd, 2nd and 4th in its three categories) but mentioned rarely: in the
  answers that cite it, its visibility is 15–31% against 54–94% for the leaders, and without a citation it is
  at most 2.2% against 11–60% for the leaders.

See `METHODS.md` for definitions, formulas and confidence intervals, and for an additional regression check
that is not relied on in the report.
