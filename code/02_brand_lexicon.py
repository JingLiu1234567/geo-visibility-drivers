# -*- coding: utf-8 -*-
"""
02_brand_lexicon.py — Brand recognition lexicon
===============================================
Output: data/brands.csv   one row per brand: key | display name | panel membership | name regex | domains
Run   : python code/02_brand_lexicon.py   (prints a full-corpus match QA: hits per brand, zero-hit warnings)

Rules
-----
1. Every name pattern is word-bounded: (?<![A-Za-z0-9]) ... (?![A-Za-z0-9]).  Stops `exa` matching `example`.
2. Matching is case-insensitive by default. Brands whose names are ordinary words force case with (?-i:...):
   - Profound / Nimble / Brave: capitalised only (lower case is an adjective / noun)
   - "Am I Cited": title case only (lower-case "am I cited" is a plain question)
   - "Scraper API" with a space: capitalised only; the one-word form scraperapi is case-insensitive
3. Category-term traps: "SERP API" (with a space) is the product category, not SerpApi;
   "Search API" is generic, not SearchApi.io; "data for SEO" is a phrase, not DataForSEO.
4. panel = the README competitor list of each category, plus cloro (in all three). Everything else is `other`:
   `other` brands never enter the main rankings but do occupy positions in an answer, so they are needed
   to compute position correctly (README: rank "among all brands named in that answer").
5. Domains are used for citation matching; sub-domains match by suffix (docs.firecrawl.dev -> firecrawl.dev).
   Verified aliases: Profound = tryprofound.com, Airefs = getairefs.com, SerpentAPI = apiserpent.com + serpentapi.com,
   Nimble = nimbleway.com, Scrunch = scrunch.com + scrunchai.com.
   SerpApi counts serpapi.com only (serpapi.cc / .org / serpapis.com look like copycat sites and are excluded).
"""
import os
import re
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "data")

S, A, T = "serp-api", "ai-tracking-api", "llm-scraping"


def b(key, name, panels, rx, doms):
    return dict(key=key, name=name, panels=";".join(panels) if panels else "other",
                name_regex=rx, domains=";".join(doms))


W1, W2 = r"(?<![A-Za-z0-9])", r"(?![A-Za-z0-9])"

BRANDS = [
 # ---- brand under study ----
 b("cloro","cloro",[S,T,A], r"cloro(?:\.dev)?", ["cloro.dev","cloro.us","cloro.mintlify.app"]),
 # ---- serp-api / llm-scraping panels ----
 b("serpapi","SerpApi",[S,T], r"serpapi(?:\.com)?", ["serpapi.com"]),
 b("dataforseo","DataForSEO",[S,T], r"dataforseo(?:\.com)?", ["dataforseo.com"]),
 b("serper","Serper",[S], r"serper(?:\.dev)?", ["serper.dev"]),
 b("brightdata","Bright Data",[S,T], r"bright\s?data", ["brightdata.com","brightdata.de","brightdata.es","brightdata.jp","brightdata.com.br","brightdata.co.kr","brightdata.mintlify.app"]),
 b("oxylabs","Oxylabs",[S,T], r"oxylabs", ["oxylabs.io","oxylabs.cn"]),
 b("searchapi","SearchApi.io",[S,T], r"searchapi(?:\.io)?", ["searchapi.io"]),
 b("scrapingdog","Scrapingdog",[S,T], r"scrapingdog", ["scrapingdog.com"]),
 b("scrapingbee","ScrapingBee",[S,T], r"scrapingbee", ["scrapingbee.com"]),
 b("scraperapi","ScraperAPI",[S,T], r"scraperapi|(?-i:Scraper API)", ["scraperapi.com","scraperapi.io"]),
 b("scrapfly","Scrapfly",[S,T], r"scrapfly", ["scrapfly.io"]),
 b("apify","Apify",[S,T], r"apify", ["apify.com","use-apify.com"]),
 b("firecrawl","Firecrawl",[S,T], r"firecrawl", ["firecrawl.dev"]),
 b("nimble","Nimble",[S,T], r"(?-i:Nimble)(?:way)?", ["nimbleway.com"]),
 b("octoparse","Octoparse",[S,T], r"octoparse", ["octoparse.com"]),
 b("olostep","Olostep",[S], r"olostep", ["olostep.com"]),
 b("serpentapi","SerpentAPI",[S,T], r"serpent\s?api", ["serpentapi.com","apiserpent.com"]),
 b("scrapebadger","ScrapeBadger",[S,T], r"scrape\s?badger", ["scrapebadger.com"]),
 b("serpbase","SerpBase",[S], r"serpbase(?:\.dev)?", ["serpbase.dev"]),
 b("openwebninja","OpenWeb Ninja",[S], r"open\s?web\s?ninja", ["openwebninja.com"]),
 b("scrapecreators","ScrapeCreators",[S], r"scrape\s?creators", ["scrapecreators.com"]),
 b("airefs","Airefs",[S,T], r"airefs", ["getairefs.com"]),
 b("crawlbase","Crawlbase",[T], r"crawlbase", ["crawlbase.com"]),
 b("scrapedo","Scrape.do",[T], r"scrape\.?do", ["scrape.do"]),
 # ---- ai-tracking-api panel ----
 b("profound","Profound",[A], r"(?-i:Profound)|tryprofound", ["tryprofound.com","profound.com"]),
 b("peec","Peec",[A], r"peec(?:\.ai)?", ["peec.ai"]),
 b("otterly","Otterly.AI",[A], r"otterly(?:\.ai)?", ["otterly.ai"]),
 b("llmpulse","LLM Pulse",[A], r"llm\s?pulse", ["llmpulse.ai"]),
 b("siftly","Siftly",[A], r"siftly(?:\.ai)?", ["siftly.ai"]),
 b("mentionsapi","MentionsAPI",[A], r"mentions\s?api", ["mentionsapi.com"]),
 b("demandsphere","DemandSphere",[A], r"demand\s?sphere", ["demandsphere.com"]),
 b("prominenceai","Prominence AI",[A], r"prominence\s?ai", ["prominenceai.io"]),
 b("geneo","Geneo",[A], r"geneo", ["geneo.app"]),
 b("scrunch","Scrunch AI",[A], r"scrunch(?:\s?ai)?", ["scrunch.com","scrunchai.com"]),
 b("rankscale","Rankscale",[A], r"rank\s?scale", ["rankscale.ai"]),
 b("evertune","Evertune",[A], r"evertune(?:\.ai)?", ["evertune.ai"]),
 b("deepsmith","DeepSmith",[A], r"deep\s?smith", ["deepsmith.ai"]),
 b("foglift","Foglift",[A], r"foglift", ["foglift.io"]),
 b("writesonic","Writesonic",[A], r"writesonic", ["writesonic.com"]),
 b("rankprompt","RankPrompt",[A], r"rank\s?prompt", ["rankprompt.com"]),
 b("ayzeo","Ayzeo",[A], r"ayzeo", ["ayzeo.com"]),
 b("amicited","AmICited",[A], r"(?-i:Am I Cited|AmICited)|amicited\.com", ["amicited.com"]),
 # ---- `other`: named in the README as "other vendors", or frequent in answers; position only ----
 b("tavily","Tavily",[], r"tavily", ["tavily.com","tavily.org"]),
 b("exa","Exa",[], r"exa(?:\.ai)?", ["exa.ai"]),
 b("semrush","Semrush",[], r"semrush", ["semrush.com"]),
 b("ahrefs","Ahrefs",[], r"ahrefs", ["ahrefs.com"]),
 b("seranking","SE Ranking",[], r"se\s?ranking", ["seranking.com"]),
 b("similarweb","Similarweb",[], r"similar\s?web", ["similarweb.com"]),
 b("bravesearch","Brave Search",[], r"(?-i:Brave)(?:\s?Search)?", ["brave.com"]),
 b("parallel","Parallel",[], r"parallel\.ai|(?-i:Parallel (?:AI|Web|Search))", ["parallel.ai"]),
 b("zenrows","ZenRows",[], r"zenrows", ["zenrows.com"]),
 b("zyte","Zyte",[], r"zyte", ["zyte.com"]),
 b("decodo","Decodo",[], r"decodo", ["decodo.com"]),
 b("smartproxy","Smartproxy",[], r"smart\s?proxy", ["smartproxy.com"]),
 b("hasdata","HasData",[], r"hasdata", ["hasdata.com"]),
 b("valueserp","ValueSERP",[], r"value\s?serp", ["valueserp.com"]),
 b("scaleserp","Scale SERP",[], r"scale\s?serp", ["scaleserp.com"]),
 b("serpstack","Serpstack",[], r"serpstack", ["serpstack.com"]),
 b("serply","Serply",[], r"serply(?:\.io)?", ["serply.io"]),
 b("serpwow","SerpWow",[], r"serpwow", ["serpwow.com"]),
 b("thordata","Thordata",[], r"thordata", ["thordata.com"]),
 b("jina","Jina AI",[], r"jina(?:\.ai)?", ["jina.ai"]),
 b("diffbot","Diffbot",[], r"diffbot", ["diffbot.com"]),
 b("browserbase","Browserbase",[], r"browserbase", ["browserbase.com"]),
 b("crawl4ai","Crawl4AI",[], r"crawl4ai", ["crawl4ai.com"]),
 b("youcom","You.com",[], r"you\.com", ["you.com"]),
 b("linkup","Linkup",[], r"linkup(?:\.so)?", ["linkup.so"]),
 b("infatica","Infatica",[], r"infatica", ["infatica.io"]),
 b("thruuu","thruuu",[], r"thruuu", ["thruuu.com"]),
 b("agentgeo","AgentGEO",[], r"agent\s?geo", ["agentgeo.org"]),
 b("zenserp","Zenserp",[], r"zenserp", ["zenserp.com"]),
 b("serplib","SerpLib",[], r"serplib", ["serplib.com"]),
 b("xcrawl","XCrawl",[], r"xcrawl", ["xcrawl.com"]),
 b("openserp","OpenSERP",[], r"openserp", ["openserp.org"]),
 b("knowledgesdk","KnowledgeSDK",[], r"knowledgesdk", ["knowledgesdk.com"]),
 # ---- `other`, added after the labelling QA: bold list-leading terms that also have a domain in the corpus.
 #      Engines (OpenAI, Google), generic tools (Playwright) and LLM gateways (OpenRouter, LiteLLM) are excluded. ----
 b("scrapeless","Scrapeless",[], r"scrapeless", ["scrapeless.com"]),
 b("scavio","Scavio",[], r"scavio", ["scavio.dev"]),
 b("scrapegraphai","ScrapeGraphAI",[], r"scrape\s?graph\s?ai", ["scrapegraphai.com"]),
 b("conductor","Conductor",[], r"(?-i:Conductor)", ["conductor.com"]),
 b("talordata","TalorData",[], r"talor\s?data", ["talordata.com"]),
 b("rankability","Rankability",[], r"rankability", ["rankability.com"]),
 b("serphouse","SERPHouse",[], r"serp\s?house", ["serphouse.com"]),
 b("tinyfish","TinyFish",[], r"tiny\s?fish", ["tinyfish.ai"]),
 b("kime","KIME",[], r"(?-i:KIME|Kime)", ["kime.ai"]),
 b("serplify","Serplify",[], r"serplify", ["serplify.io"]),
 b("promptrush","PromptRush",[], r"prompt\s?rush", ["promptrush.ai"]),
 b("brightedge","BrightEdge",[], r"bright\s?edge", ["brightedge.com"]),
 b("outscraper","Outscraper",[], r"outscraper", ["outscraper.com"]),
 b("nightwatch","Nightwatch",[], r"(?-i:Nightwatch)", ["nightwatch.io"]),
 b("athenahq","AthenaHQ",[], r"athena\s?hq", ["athenahq.ai"]),
 b("genrank","GenRank",[], r"gen\s?rank", ["genrank.io"]),
 b("searlo","Searlo",[], r"searlo", ["searlo.tech"]),
 b("contextdev","Context.dev",[], r"context\.dev", ["context.dev"]),
 b("sellm","Sellm",[], r"sellm", ["sellm.io"]),
 b("alterlab","AlterLab",[], r"alter\s?lab", ["alterlab.io"]),
 b("cognizo","Cognizo",[], r"cognizo", ["cognizo.ai"]),
 b("ziptie","ZipTie",[], r"zip\s?tie", ["ziptie.dev"]),
 b("searchcans","SearchCans",[], r"search\s?cans", ["searchcans.com"]),
 b("brand24","Brand24",[], r"brand\s?24", ["brand24.com"]),
 b("aiclicks","AIclicks",[], r"ai\s?clicks", ["aiclicks.io"]),
 b("workduo","WorkDuo",[], r"work\s?duo", ["workduo.ai"]),
 b("spidra","Spidra",[], r"spidra", ["spidra.io"]),
 b("keywordcom","Keyword.com",[], r"keyword\.com", ["keyword.com"]),
 b("trajectdata","Traject Data",[], r"traject\s?data", ["trajectdata.com"]),
 b("kadoa","Kadoa",[], r"kadoa", ["kadoa.com"]),
 b("promptwatch","Promptwatch",[], r"prompt\s?watch", ["promptwatch.com"]),
 b("finseo","Finseo",[], r"finseo", ["finseo.ai"]),
 b("citadex","Citadex",[], r"citadex", ["citadex.io"]),
 b("beamtrace","Beamtrace",[], r"beam\s?trace", ["beamtrace.com"]),
 b("wellows","Wellows",[], r"wellows", ["wellows.com"]),
 b("mentionscout","MentionScout",[], r"mention\s?scout", ["mentionscout.com"]),
 b("knowatoa","Knowatoa",[], r"knowatoa", ["knowatoa.com"]),
 b("jetoctopus","JetOctopus",[], r"jet\s?octopus", ["jetoctopus.com"]),
 b("citlyze","Citlyze",[], r"citlyze", ["citlyze.com"]),
 b("bringits","Bringits",[], r"bringits", ["bringits.com"]),
 b("serpsearch","SERP Search",[], r"(?-i:SERP Search)|serpsearch\.com", ["serpsearch.com"]),
 b("browserless","Browserless",[], r"browserless", ["browserless.io"]),
 b("brandtrace","Brandtrace",[], r"brandtrace", ["brandtrace.net"]),
 b("edenai","Eden AI",[], r"eden\s?ai", ["edenai.co"]),
 b("shifter","Shifter",[], r"(?-i:Shifter)", ["shifter.io"]),
 b("scrapellm","ScrapeLLM",[], r"scrape\s?llm", ["scrapellm.com"]),
 b("scrapeinsight","ScrapeInsight",[], r"scrape\s?insight", ["scrapeinsight.com"]),
 b("keirolabs","Keirolabs",[], r"keiro\s?labs", ["keirolabs.cloud"]),
 b("antsdata","AntsData",[], r"ants\s?data", ["antsdata.com"]),
 b("aeovision","AEO Vision",[], r"aeo\s?vision", ["aeovision.ai"]),
 b("trustablelabs","Trustable Labs",[], r"trustable\s?labs", ["trustablelabs.com"]),
 b("measurellm","MeasureLLM",[], r"measure\s?llm", ["measurellm.com"]),
 b("frizerly","Frizerly",[], r"frizerly", ["frizerly.com"]),
 b("frase","Frase",[], r"(?-i:Frase)", ["frase.io"]),
 b("meltwater","Meltwater",[], r"meltwater", ["meltwater.com"]),
 b("brandwatch","Brandwatch",[], r"brandwatch", ["brandwatch.com"]),
 b("talkwalker","Talkwalker",[], r"talkwalker", ["talkwalker.com"]),
 b("muckrack","Muck Rack",[], r"muck\s?rack", ["muckrack.com"]),
 b("visualping","Visualping",[], r"visualping", ["visualping.io"]),
 b("fetchserp","FetchSERP",[], r"fetch\s?serp", ["fetchserp.com"]),
 b("browseract","BrowserAct",[], r"browseract", ["browseract.com"]),
 b("authoritas","Authoritas",[], r"authoritas", ["authoritas.com"]),
 b("opttab","Opttab",[], r"opttab", ["opttab.com"]),
 b("aisearchapi","AI Search API (aisearchapi.dev)",[], r"aisearchapi(?:\.dev)?", ["aisearchapi.dev"]),
 b("spidercloud","Spider.cloud",[], r"spider\.cloud", ["spider.cloud"]),
]

lex = pd.DataFrame(BRANDS)
assert lex.key.is_unique
lex.to_csv(os.path.join(D, "brands.csv"), index=False, encoding="utf-8-sig")

# ---------------- QA: match the whole corpus ----------------
df = pd.read_parquet(os.path.join(D, "clean.parquet"))
m = df[(df.window == "core") & (df.engine != "CLAUDE") & df.markdown.notna()]
prose = m.markdown.str.replace(r"\[([^\]]*)\]\([^)]*\)", r"\1", regex=True)   # strip link markup, keep anchor text (QA only)
text = "\n".join(prose)

print(f"lexicon: {len(lex)} brands (panel {sum(lex.panels != 'other')} + other {sum(lex.panels == 'other')}) -> data/brands.csv")
print(f"QA corpus: {len(m)} answers with prose in the core window\n")
rows = []
for r in BRANDS:
    pat = re.compile(W1 + "(?:" + r["name_regex"] + ")" + W2, re.I)
    hits = pat.findall(text)
    top = pd.Series(hits).value_counts().head(3).to_dict() if hits else {}
    rows.append((r["key"], r["panels"], len(hits), top))
qa = pd.DataFrame(rows, columns=["key", "panels", "hits", "top_forms"]).sort_values("hits", ascending=False)
pd.set_option("display.width", 200); pd.set_option("display.max_colwidth", 70)
print(qa.to_string(index=False))
zero = qa[(qa.hits == 0) & (qa.panels != "other")]
print(f"\npanel brands with zero hits (need manual review): {list(zero.key)}")
