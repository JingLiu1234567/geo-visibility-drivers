# Methods: how the numbers were computed

This note documents the mathematical mechanisms behind the conclusions in the report, and maps each
conclusion to the computation, script and output file that produced it.

## 1. Unit of analysis and populations

After preprocessing (`01_preprocess.py`) and labelling (`03_extract.py`), the analysis table has one row per
(answer, brand) pair. The main analysis keeps:

- window = core (Aug 11–20; the same 170 prompts ran on every engine every day),
- the five engines that return prose (ChatGPT, Perplexity, Gemini, Copilot, AI Mode),
- answers with text (8,384),
- for each answer, the competitor panel of its category (motion) plus cloro: 22 brands in serp-api,
  19 in llm-scraping, 19 in ai-tracking-api.

This gives **167,276 rows**. Per brand and motion: n = 2,660 (serp-api), 2,716 (llm-scraping), 3,008 (ai-tracking-api).

Every row carries five binary labels — `mentioned`, `position` (rank), `cited`, `listed_option`,
`named_in_prompt` — and the answer's attributes: engine, motion, intent, topic, number of brands named,
number of sources, answer length.

## 2. Metrics are conditional rates

Every percentage in the report is the mean of a binary label over a defined set of rows:

| metric | formula | README definition |
|---|---|---|
| visibility | P(mentioned = 1) within a (motion, engine, brand) group | Visibility |
| first-position rate | P(position = 1 and named_in_prompt = 0) | Position = 1, minus the target effect |
| cited rate | P(cited = 1) | Citation |

Denominators are all answers with prose in the group (answers that name no brand stay in the denominator).

## 3. "What drives X" = contrasts of conditional probabilities

For each candidate factor the rows are split on that factor and the outcome rate is compared on each side:

- Visibility driver (report §5.1): P(mentioned | cited) = 43.9% vs P(mentioned | not cited) = 4.8%;
  lift = 43.9 / 4.8 = 9.1. Repeated within each engine and each motion (`out/drv_citation_by_engine.csv`).
- Brand memory (report §5.2): the same contrast per brand (`out/drv_brand_conversion.csv`).
  P(mentioned | not cited) measures how often a brand is named when none of its pages were retrieved.
- First position driver (report §5.3): among mentioned rows, P(position = 1 | brand is the first-ranked source)
  = 71.8% vs 20.9% otherwise.

## 4. Confounding control: fixed-effects regressions (`05_drivers.py`, section F)

A raw split can be confounded — strong brands are both cited more and mentioned more, and some prompts elicit
brand names while others never do. Two regressions with the same right-hand side isolate each factor:

- **Logit** with brand, engine, topic and intent fixed effects. Reported as odds ratios.
- **Linear probability model (OLS)** with brand, engine and **prompt** fixed effects, so that every comparison is
  made within the same question. Reported in percentage points. (Prompt fixed effects cannot be used in the
  logit because some prompts never name any brand, which causes perfect separation.)

Rows for brands never mentioned in the sample are dropped from the logit (they carry no information).

**Non-independence.** Each prompt is asked on each engine on ten days, so answers are repeated measurements.
All standard errors are clustered by prompt × engine (1,190 clusters).

### Regression results

| outcome | model | n | cited | first source | named in prompt | brands named (per +1) | listed option |
|---|---|---|---|---|---|---|---|
| mentioned | logit, brand/engine/topic/intent FE | 167,276 | OR 41.2*** | – | OR 97.4*** | OR 1.57*** | – |
| mentioned | LPM, brand/engine/prompt FE | 167,276 | +32.1 pp*** | – | +46.3 pp*** | +2.7 pp*** | – |
| first position \| mentioned | logit, brand/engine/topic/intent FE | 13,451 | OR 1.45*** | OR 11.0*** | OR 691*** | OR 0.80*** | OR 1.04 (p = 0.60) |
| first position \| mentioned | LPM, brand/engine/prompt FE | 13,451 | +6.1 pp*** | +40.3 pp*** | +81.1 pp*** | −2.2 pp*** | +1.5 pp (p = 0.17) |
| cloro mentioned | logit, engine/motion/intent | 8,384 | OR 57.4*** | – | – | OR 1.35*** | – |

\*\*\* p < 0.001 (cluster-robust). Pseudo-R² 0.53 (mentioned), 0.35 (first position). Other terms in the
mention model: informational intent OR 0.80 (p = 0.008); log sources OR 0.82 (p < 0.001, more sources dilute each
brand); log length OR 1.19 (p = 0.006). Full output: `out/regression.txt`.

Effect of `cited` by engine (LPM with prompt FE, percentage points, all p < 0.001):
Copilot +50.5 · Gemini +49.8 · ChatGPT +42.8 · AI Mode +31.9 · Perplexity +14.8.

## 5. Uncertainty: Wilson 95% confidence intervals

| brand (motion) | n cited | P(mention \| cited) | n not cited | P(mention \| not cited) |
|---|---|---|---|---|
| SerpApi (serp-api) | 1,218 | 93.8% [92.3–95.1] | 1,442 | 60.0% [57.4–62.5] |
| Bright Data (llm-scraping) | 500 | 65.0% [60.7–69.1] | 2,216 | 10.8% [9.6–12.1] |
| Profound (ai-tracking-api) | 288 | 53.5% [47.7–59.1] | 2,720 | 17.8% [16.4–19.3] |
| cloro (serp-api) | 792 | 14.8% [12.5–17.4] | 1,868 | 0.2% [0.1–0.5] |
| cloro (llm-scraping) | 452 | 26.5% [22.7–30.8] | 2,264 | 2.2% [1.6–2.8] |
| cloro (ai-tracking-api) | 238 | 31.5% [25.9–37.7] | 2,770 | 0.4% [0.2–0.7] |
| all brands pooled | 13,744 | 43.9% [43.1–44.8] | 153,532 | 4.8% [4.7–4.9] |
| first position \| mentioned, first source | 1,494 | 71.8% [69.4–74.0] | 11,968 (not first) | 20.9% [20.2–21.6] |

None of the cloro–leader intervals overlap.

## 6. Decomposition of cloro's visibility (report §4.4)

Over the 850 (prompt, engine) pairs of the core window (`out/stability_cloro.csv`):

visibility = coverage × conditional stability = 14.5% × 31% ≈ 4.5%,

where coverage is the share of pairs that mention cloro on at least one of the ten days (123 of 850) and
conditional stability is the mean share of days mentioned within those pairs. Only 17 pairs mention cloro on
more than 70% of days (Gemini 9, Copilot 6).

## 7. Robustness checks

1. **Link-anchor policy** (labelling). Mentions were recomputed under a stricter rule (delete every link, anchor
   text included) and a looser one (keep all anchor text). The strict rule moves cloro's visibility by at most
   0.8 points per engine; the loose rule inflates Gemini to 16% by counting page titles such as
   "cloro.dev Review" as mentions. The middle rule (keep anchors of ≤ 3 words) is used.
2. **Listed-option recount** (report §4.5). Visibility was recomputed counting only mentions where the brand is
   presented as an option (bold / list item / table row). The top three brands and cloro's rank are unchanged
   in every motion; ranks move only among lower-visibility brands.
3. **Two model families** (section 4). Logit and LPM agree on sign and significance for every term.

## 8. Conclusion → computation map

| conclusion in the report | computation | script / output |
|---|---|---|
| cloro is cited at top-tier rates but mentioned rarely (§4.3) | cited rate vs visibility per motion | `04_describe.py` → `leaderboard_pooled.csv` |
| citation is the strongest driver of visibility (§5.1) | P(mention \| cited) vs P(mention \| not cited); FE regressions | `05_drivers.py` A, F → `drv_citation_by_engine.csv`, `regression.txt` |
| leaders keep visibility without citation, cloro does not (§5.2) | per-brand conditional rates + Wilson CIs | `05_drivers.py` B → `drv_brand_conversion.csv` |
| first position is driven by citation order (§5.3) | P(#1 \| mentioned, first source) contrast; FE regressions on mentioned rows | `05_drivers.py` E, F |
| cloro's visibility is unstable across days (§4.4) | coverage × stability decomposition | `04_describe.py` D → `stability_cloro.csv` |
| rankings are robust to stricter counting (§4.5) | listed-option recount | `04_describe.py` (`listed_rate`) |
