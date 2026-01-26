# app.py — Erfocentrum prototype (Shiny for Python) — single search topbar + AI button after search
from shiny import App, reactive, render, ui
from openai import OpenAI
from pathlib import Path
import pandas as pd
import numpy as np
import re
import json
import os
import unicodedata
from difflib import get_close_matches
from openai import RateLimitError

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
DATA_DIR = Path("data")
APP_DIR = Path(__file__).parent

# -------------------------
# Helpers
# -------------------------
def nz(x: str) -> bool:
    return isinstance(x, str) and len(x.strip()) > 0

def safe_id(x: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", str(x))

def norm_text(s: str) -> str:
    """Lowercase + accents weg + non-alnum -> spaties + trim."""
    if s is None:
        return ""
    s = str(s).lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^a-z0-9à-ÿ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def split_terms(s: str) -> list[str]:
    """Split aliases/synonyms op veel mogelijke separators."""
    if not s:
        return []
    parts = re.split(r"[|,;/\n\r\t]+", str(s))
    out = []
    for p in parts:
        p = norm_text(p)
        if p:
            out.append(p)
    return out

def fuzzy_suggest(query: str, vocab_norm: list[str], n=1, cutoff=0.75):
    qn = norm_text(query)
    if not qn:
        return []
    return get_close_matches(qn, vocab_norm, n=n, cutoff=cutoff)

def local_related_terms(query: str, syn_df: pd.DataFrame, max_n=6):
    qn = norm_text(query)
    toks = [t for t in qn.split() if len(t) >= 3]
    if syn_df is None or len(syn_df) == 0:
        return []
    syn_map = dict(zip(syn_df["term"].tolist(), syn_df["synonyms"].tolist()))
    out = []
    for t in toks:
        if t in syn_map:
            out.extend(split_terms(syn_map[t]))
    # uniek + max
    out = [x for x in dict.fromkeys(out) if x and x not in toks]
    return out[:max_n]

def extract_glossary_hits(text: str, glossary_df: pd.DataFrame, max_n: int = 6):
    if not nz(text):
        return glossary_df.iloc[0:0].copy()
    t = text.lower()
    g = glossary_df.copy()
    g["term_low"] = g["term"].fillna("").astype(str).str.lower()
    hits = g[g["term_low"].apply(lambda term: nz(term) and term in t)]
    return hits[["term", "uitleg"]].head(max_n)

def contains_wordish(series: pd.Series, term: str) -> pd.Series:
    """
    Match op woord-begin + eventuele letters erachter.
    Voorbeeld: 'spierziekte' matcht ook 'spierziekten' en 'spierziekteonderzoek'.
    """
    if not term:
        return series.astype(str).str.contains(r"a^")  # altijd False
    pat = r"\b" + re.escape(term) + r"\w*\b"
    return series.astype(str).str.contains(pat, regex=True, na=False)

def llm_to_dutch_query(user_text: str) -> dict:
    user_text = (user_text or "").strip()
    if not user_text:
        return {"language": "", "dutch_query": "", "keywords": []}

    prompt = f"""
Je bent een medische informatie-navigator voor een Nederlandse site over erfelijkheid en gezondheid.
Zet de gebruikerstekst om naar een korte Nederlandse zoekopdracht + trefwoorden.

Regels:
- Detecteer de taal.
- Vertaal naar correct Nederlands (geen Engels).
- Geef 3 tot 8 Nederlandse trefwoorden/synoniemen die helpen om artikelen te vinden.
- Houd de zoekopdracht kort (max 3 woorden).
- Output MUST be valid JSON met exact de keys: language, dutch_query, keywords.

Gebruikerstekst:
{user_text}
""".strip()

    try:
        resp = client.responses.create(
            model="gpt-4.1-mini",
            input=prompt,
        )
        txt = resp.output_text.strip()
    except Exception:
        # ✅ Nooit de zoekflow breken als OpenAI faalt
        return {"language": "", "dutch_query": user_text, "keywords": [user_text]}

    m = re.search(r"\{.*\}", txt, flags=re.S)
    if m:
        txt = m.group(0)

    try:
        data = json.loads(txt)
    except Exception:
        return {"language": "", "dutch_query": user_text, "keywords": [user_text]}

    dq = str(data.get("dutch_query", "")).strip()
    kws = data.get("keywords", [])
    if not isinstance(kws, list):
        kws = []
    kws = [str(x).strip().lower() for x in kws if str(x).strip()]
    return {
        "language": str(data.get("language", "")).strip(),
        "dutch_query": dq or user_text,
        "keywords": list(dict.fromkeys(kws))[:12],
    }

def tokenize_nl(text: str):
    x = (text or "").lower()
    x = re.sub(r"[^a-z0-9à-ÿ\s-]", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    if not x:
        return []
    toks = [t for t in x.split(" ") if len(t) >= 3]
    stop = {
        "een", "het", "dat", "wat", "hoe", "waar", "waarom", "met", "naar", "voor", "van", "mijn", "jouw", "jij", "ik",
        "ook", "niet", "wel", "kun", "kan", "wil", "wilt", "ben", "zijn", "was", "dit", "deze", "die", "dus", "als", "bij",
        "nog", "is", "heeft", "heb", "had", "maar", "of", "en"
    }
    toks = [t for t in toks if t not in stop]
    return sorted(set(toks))

def score_docs_for_prompt(prompt: str, content_df: pd.DataFrame, themes_df: pd.DataFrame, syn_df: pd.DataFrame, top_n=5):
    p = (prompt or "").strip()
    if not p:
        return content_df.iloc[0:0].copy()

    toks = tokenize_nl(p)

    # add synonyms for tokens
    syn_add = []
    if syn_df is not None and len(syn_df) and len(toks):
        syn_map = dict(zip(syn_df["term"], syn_df["synonyms"]))
        for t in toks:
            if t in syn_map:
                extra = [x.strip().lower() for x in str(syn_map[t]).split(",")]
                syn_add.extend([e for e in extra if e])
    toks = sorted(set(toks + syn_add))

    theme_hits = []
    if themes_df is not None and len(themes_df):
        for lab in themes_df["label"].fillna("").astype(str):
            if lab and lab.lower() in p.lower():
                theme_hits.append(lab)
    theme_hits = sorted(set(theme_hits))

    df = content_df.copy()
    df["t_title"] = df["title"].str.lower()
    df["t_short"] = df["short"].str.lower()
    df["t_alias"] = df["aliases"].str.lower()
    df["t_cat"] = df["category"].str.lower()
    df["t_all"] = df["t_title"] + " | " + df["t_short"] + " | " + df["t_alias"] + " | " + df["t_cat"]

    score = np.zeros(len(df), dtype=int)
    for t in toks:
        score += 3 * df["t_title"].str.contains(re.escape(t), regex=True).astype(int)
        score += 2 * df["t_short"].str.contains(re.escape(t), regex=True).astype(int)
        score += 2 * df["t_alias"].str.contains(re.escape(t), regex=True).astype(int)
        score += 1 * df["t_cat"].str.contains(re.escape(t), regex=True).astype(int)

    for th in theme_hits:
        th_low = th.lower()
        score += 2 * df["t_all"].str.contains(re.escape(th_low), regex=True).astype(int)

    df["score"] = score
    out = df[df["score"] > 0].sort_values("score", ascending=False)
    cols = ["doc_id", "title", "category", "url", "short", "long", "next_step", "disclaimer", "aliases", "score"]
    return out[cols].head(top_n)

def next_question_for_prompt(prompt: str):
    p = (prompt or "").lower()
    if not p:
        return "Kun je in één zin vertellen wat je wilt weten?"
    if re.search(r"zwanger|kinderwens|ivf|pgd|pgt", p):
        return "Gaat het om kinderwens (vooraf testen) of om een uitslag die je al hebt?"
    if re.search(r"uitslag|resultaat|test|dna", p):
        return "Weet je om welk type DNA-onderzoek het gaat (dragerschap, diagnostiek, prenataal)?"
    if re.search(r"familie|erfelijk|ouders|broer|zus", p):
        return "Gaat het om ziekte in de familie, of wil je weten of jij drager bent?"
    return "Wil je vooral uitleg, of vooral wat je nu het beste kunt doen (volgende stap)?"


def llm_rerank(query: str, cand_df: pd.DataFrame, k: int = 5):
    if cand_df is None or len(cand_df) == 0:
        return {"top_ids": [], "did_you_mean": None, "extra_synonyms": [], "reasoning": []}

    items = []
    for _, r in cand_df.iterrows():
        items.append({
            "doc_id": str(r["doc_id"]),
            "title": str(r["title"]),
            "category": str(r["category"]),
            "short": str(r["short"]),
            "aliases": str(r["aliases"])
        })

    system = (
        "Je bent een Nederlandse medische informatie-navigator. "
        "Je mag alleen doc_id's kiezen die in de kandidatenlijst staan. "
        "Geef JSON terug. Geen extra tekst."
    )

    user = {
        "query": query,
        "candidates": items,
        "instructions": {
            "task": "Kies de beste doc_id matches voor de query.",
            "return": {
                "top_ids": f"lijst met max {k} doc_id strings, beste eerst",
                "did_you_mean": "string of null (als er een duidelijke correctie is)",
                "extra_synonyms": "lijst met NL synoniemen/verwante termen (max 6)",
                "reasoning": "lijst met korte redenen per gekozen doc_id (max 1 zin per item)"
            }
        }
    }

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user, ensure_ascii=False)}
            ],
            temperature=0.1
        )
    except RateLimitError:
        # Geen crash: gewoon geen LLM-hints/rerank
        return {"top_ids": [], "did_you_mean": None, "extra_synonyms": [], "reasoning": []}

    txt = resp.choices[0].message.content.strip()
    try:
        data = json.loads(txt)
    except Exception:
        data = {"top_ids": [], "did_you_mean": None, "extra_synonyms": [], "reasoning": []}

    valid_ids = set(cand_df["doc_id"].astype(str).tolist())
    top_ids = [x for x in data.get("top_ids", []) if str(x) in valid_ids][:k]

    return {
        "top_ids": top_ids,
        "did_you_mean": data.get("did_you_mean", None),
        "extra_synonyms": (data.get("extra_synonyms", []) or [])[:6],
        "reasoning": (data.get("reasoning", []) or [])[:k]
    }

# -------------------------
# Load data
# -------------------------
content_raw = pd.read_csv(DATA_DIR / "dsp_dataset_erfocentrum_v1.csv")
content_ui = (
    content_raw.rename(columns={"name": "title", "short_text_simple": "short", "long_text_complex": "long"})
    .loc[:, ["doc_id", "title", "category", "url", "short", "long", "next_step", "disclaimer", "aliases"]]
    .drop_duplicates(subset=["doc_id"], keep="first")
)

content_ui["category"] = content_ui["category"].fillna("").replace("", "Overig")
for c in ["doc_id", "title", "category", "url", "short", "long", "next_step", "disclaimer", "aliases"]:
    content_ui[c] = content_ui[c].fillna("").astype(str)

themes = pd.read_csv(DATA_DIR / "themes.csv")
hl_blocks = pd.read_csv(DATA_DIR / "hl_support_blocks.csv")
glossary = pd.read_csv(DATA_DIR / "glossary.csv")
syn = pd.read_csv(DATA_DIR / "synonyms.csv")

syn["term"] = syn["term"].fillna("").astype(str).str.strip().str.lower()
syn["synonyms"] = syn["synonyms"].fillna("").astype(str).str.strip().str.lower()

alias_terms = []
for a in content_ui["aliases"].fillna("").astype(str).tolist():
    if a.strip():
        alias_terms.extend([x.strip() for x in re.split(r"\s*[\|,;]\s*", a) if x.strip()])

vocab = sorted(set(
    content_ui["title"].tolist()
    + content_ui["category"].tolist()
    + alias_terms
    + syn["term"].tolist()
))
vocab_choices = [""] + vocab

# Normalized search columns
content_ui["title_n"] = content_ui["title"].apply(norm_text)
content_ui["short_n"] = content_ui["short"].apply(norm_text)
content_ui["long_n"] = content_ui["long"].apply(norm_text)
content_ui["cat_n"] = content_ui["category"].apply(norm_text)
content_ui["aliases_n"] = content_ui["aliases"].apply(lambda x: " ".join(split_terms(x)))
content_ui["all_n"] = (
    content_ui["title_n"] + " " +
    content_ui["aliases_n"] + " " +
    content_ui["short_n"] + " " +
    content_ui["long_n"] + " " +
    content_ui["cat_n"]
).str.strip()

# vocab voor suggesties/spelling
vocab_terms = set()
vocab_terms.update(content_ui["title_n"].tolist())
vocab_terms.update(content_ui["cat_n"].tolist())
for a in content_ui["aliases"].tolist():
    for t in split_terms(a):
        vocab_terms.add(t)
if syn is not None and len(syn) > 0:
    for t in syn["term"].tolist():
        vocab_terms.add(norm_text(t))
    for s in syn["synonyms"].tolist():
        for t in split_terms(s):
            vocab_terms.add(t)

VOCAB_NORM = sorted([t for t in vocab_terms if t])

# -------------------------
# UI
# -------------------------
app_ui = ui.page_fluid(
    ui.include_css("www/styles.css"),
    ui.tags.head(
        # JS helpers
        ui.tags.script("""
window.erfoSetInput = function(id, value){
  if (window.Shiny && Shiny.setInputValue){
    Shiny.setInputValue(id, value, {priority: "event"});
  }
}

window.erfoClick = function(buttonId){
  var btn = document.getElementById(buttonId);
  if (btn) btn.click();
}

window.erfoBindEnterToButton = function(selectId, buttonId){
  function tryBind(){
    var inputEl = document.getElementById(selectId + "-selectized");
    if (!inputEl){
      var container = document.getElementById(selectId);
      if (container){
        inputEl = container.querySelector(".selectize-input input");
      }
    }
    if (!inputEl) return false;

    inputEl.addEventListener("keydown", function(e){
      if (e.key === "Enter"){
        e.preventDefault();
        e.stopPropagation();
        window.erfoClick(buttonId);
      }
    });
    return true;
  }

  if (tryBind()) return;

  var tries = 0;
  var timer = setInterval(function(){
    tries += 1;
    if (tryBind() || tries > 20){
      clearInterval(timer);
    }
  }, 250);
}

document.addEventListener("DOMContentLoaded", function(){
  window.erfoBindEnterToButton("q", "q_go");
});
        """),
    ),
    ui.output_ui("page")
)

# -------------------------
# Server
# -------------------------
def server(input, output, session):
    # We starten direct in search
    current_page = reactive.Value("search")  # search | detail
    selected_id = reactive.Value(None)

    # "Submit" gedrag: pas na Zoek/Enter komt er een committed query
    committed_q = reactive.Value("")

    # AI state
    bot_history = reactive.Value([])  # list[dict]
    bot_recs = reactive.Value(content_ui.iloc[0:0].copy())

    # LLM rerank state
    llm_rank = reactive.Value({"top_ids": [], "did_you_mean": None, "extra_synonyms": [], "reasoning": []})
    llm_cache = reactive.Value({})  # {query: rank_result}
    # -------------------------
    # Pages
    # -------------------------
    def search_page():
        return ui.div(
            {"class": "erfo-wrap"},
            ui.div(
                {"class": "erfo-topbar"},
                ui.div({"class": "erfo-logo"}, ui.img(src="erfo_logo.png", height="44px")),
                ui.div(
                    {"class": "erfo-search"},
                    ui.input_selectize(
                        "q",
                        None,
                        choices=vocab_choices,
                        selected="",
                        options={
                            "placeholder": "Zoek op een ziekte of onderwerp…",
                            "create": True,
                            "openOnFocus": True,
                            "allowEmptyOption": True
                        },
                    ),
                    ui.input_action_button("q_go", "Zoek", class_="btn-erfo-search"),
                ),
                ui.div(
                    {"class": "erfo-help"},
                    ui.tags.a(
                        "Vragen? Mail de Erfolijn",
                        href="mailto:info@erfocentrum.nl",
                        class_="btn btn-erfo-help",
                    ),
                ),
            ),
            ui.output_ui("home_news_block"),

            # AI-knop verschijnt pas na zoeken
            ui.output_ui("ai_help_section"),

            ui.div(
                {"class": "erfo-results"},
                ui.div(
                    {"class": "erfo-results-header"},
                    ui.div(
                        {"class": "erfo-results-titlewrap"},
                        ui.h2({"class": "erfo-h2"}, "Resultaten"),
                        ui.output_ui("results_count"),
                    ),
                    ui.div(
                        {"class": "erfo-hints"},
                        ui.output_ui("spell_hint_results"),
                        ui.output_ui("llm_hint"),
                    ),
                ),
                ui.div({"class": "erfo-results-list"}, ui.output_ui("results_cards")),
            ),
        )
    def detail_page():
        return ui.div(
            {"class": "erfo-wrap"},
            ui.div(
                {"class": "erfo-topbar"},
                ui.div({"class": "erfo-logo"}, ui.img(src="erfo_logo.png", height="44px")),
                ui.div(
                    {"class": "erfo-search"},
                    ui.input_selectize(
                        "q",
                        None,
                        choices=vocab_choices,
                        selected="",
                        options={
                            "placeholder": "Zoek op een ziekte of onderwerp…",
                            "create": True,
                            "openOnFocus": True,
                            "allowEmptyOption": True
                        },
                    ),
                    ui.input_action_button("q_go", "Zoek", class_="btn-erfo-search"),
                ),
                ui.div(
                    {"class": "erfo-help"},
                    ui.tags.a(
                        "Vragen? Mail de Erfolijn",
                        href="mailto:info@erfocentrum.nl",
                        class_="btn btn-erfo-help",
                    ),
                ),
            ),
            ui.card(
                ui.h2(ui.output_text("detail_title")),
                ui.p({"class": "meta"}, ui.output_text("detail_meta")),
                ui.output_ui("detail_layer"),
                ui.hr(),
                ui.p(ui.output_text("detail_short")),
                ui.tags.details(
                    ui.tags.summary("Lees volledige tekst"),
                    ui.p(ui.output_text("detail_long")),
                ),
                ui.hr(),
                ui.h4("Volgende stap"),
                ui.p(ui.output_text("detail_next")),
                ui.h4("Disclaimer"),
                ui.p(ui.output_text("detail_disclaimer")),
                ui.br(),
                ui.input_action_button("back_to_search", "← Terug naar resultaten", class_="btn-outline-secondary"),
            )
        )

    @output
    @render.ui
    def page():
        p = current_page.get()
        if p == "detail":
            return detail_page()
        return search_page()

    @reactive.effect
    @reactive.event(input["back_to_search"])
    def _back_to_search():
        current_page.set("search")

    @reactive.effect
    def _clear_search_on_start():
        try:
            ui.update_selectize("q", selected="")
        except Exception:
            pass
    # -------------------------
    # Commit query on Zoek (button or Enter)
    # -------------------------
    @reactive.effect
    @reactive.event(input["q_go"])
    def _commit_query():
        raw = (input["q"]() or "").strip()
        if not raw:
            committed_q.set("")
            llm_rank.set({"top_ids": [], "did_you_mean": None, "extra_synonyms": [], "reasoning": []})
            return

        # Optioneel: vertaal/normaliseer naar NL (handig voor Engelstalige input)
        mapped = llm_to_dutch_query(raw)
        q_nl = (mapped.get("dutch_query") or raw).strip()
        q_nl = norm_text(q_nl)
        q_nl = re.sub(r"^(wat is|wat zijn|hoe werkt|hoe kan|waarom|wanneer)\s+", "", q_nl).strip()

        # ✅ maak van vraagzin weer een zoekterm
        q_nl = norm_text(q_nl)
        q_nl = re.sub(r"^(wat is|wat zijn|hoe werkt|hoe kan|waarom|wanneer)\s+", "", q_nl).strip()
        
        committed_q.set(q_nl)
        print("COMMITTED:", q_nl)

        # Toon de "gecommit" query ook in het veld (zodat het voelt alsof je écht gezocht hebt)
        try:
            ui.update_selectize("q", selected=q_nl)
        except Exception:
            pass

        # rerank alleen na commit
        cand = candidates()

        # Als we al genoeg resultaten hebben, skip LLM rerank
        if cand is not None and len(cand) >= 8:
            llm_rank.set({"top_ids": [], "did_you_mean": None, "extra_synonyms": [], "reasoning": []})
            return

        # ---- FIX 2: cache per zoekterm ----
        cache = llm_cache.get()
        key = q_nl.lower().strip()

        if key in cache:
            # Gebruik cached resultaat
            llm_rank.set(cache[key])
        else:
            # Nieuwe LLM call
            r = llm_rerank(q_nl, cand, k=5)
            cache[key] = r
            llm_cache.set(cache)
            llm_rank.set(r)

    # -------------------------
    # CORE SEARCH LOGIC (uses committed_q)
    # -------------------------
    @reactive.calc
    def filtered():
        df = content_ui.copy()

        q_raw = (committed_q.get() or "").strip()
        qn = norm_text(q_raw)
        if not qn:
            return df.iloc[0:0].copy()

        stop = {"wat","hoe","waar","wanneer","waarom","is","zijn","de","het","een"}
        tokens = [t for t in qn.split(" ") if len(t) >= 3 and t not in stop]
        
        # ✅ Multi-word queries: AND-filter
        if len(tokens) >= 2:
            mask = np.ones(len(df), dtype=bool)
            for t in tokens:
                t_esc = re.escape(t)
                mask &= (
                    df["title_n"].str.contains(t_esc, regex=True) |
                    df["aliases_n"].str.contains(t_esc, regex=True)
                )

            df = df[mask].copy()

            if len(df) == 0:
                return df.iloc[0:0].copy()
        
        expanded = set(tokens)
        if syn is not None and len(syn) > 0:
            syn_map = dict(zip(syn["term"].tolist(), syn["synonyms"].tolist()))
            for t in list(expanded):
                if t in syn_map:
                    for extra in split_terms(syn_map[t]):
                        expanded.add(extra)

        # theme labels (optioneel)
        if themes is not None and len(themes) > 0:
            for lab in themes["label"].fillna("").astype(str).tolist():
                ln = norm_text(lab)
                if ln and (ln in qn or qn in ln):
                    expanded.add(ln)

        expanded = sorted([t for t in expanded if t])

        score = np.zeros(len(df), dtype=int)

        # exact phrase boost
        score += 8 * df["title_n"].str.contains(re.escape(qn), regex=True).astype(int)
        score += 6 * df["aliases_n"].str.contains(re.escape(qn), regex=True).astype(int)

        for t in expanded:
            t_esc = re.escape(t)
            score += 5 * df["title_n"].str.contains(t_esc, regex=True).astype(int)
            score += 4 * df["aliases_n"].str.contains(t_esc, regex=True).astype(int)
            score += 2 * df["short_n"].str.contains(t_esc, regex=True).astype(int)
            score += 1 * df["long_n"].str.contains(t_esc, regex=True).astype(int)
            score += 1 * df["cat_n"].str.contains(t_esc, regex=True).astype(int)

        df["score"] = score
        hits = df[df["score"] > 0].sort_values(["score", "title"], ascending=[False, True])

        # ✅ ALS ER HITS ZIJN: teruggeven
        if len(hits) > 0:
            return hits.drop(columns=["score"], errors="ignore").copy()

        # -----------------------------
        # ✅ GEEN HITS -> punt 4 gedrag
        #   - multi-word: GEEN fuzzy (dus "angelina jolie" = 0 resultaten)
        #   - single-word: WEL fuzzy (typo-correctie)
        # -----------------------------
        is_multiword = (len(tokens) >= 2)
        if is_multiword:
            return df.iloc[0:0].copy()

        sugg = fuzzy_suggest(qn, VOCAB_NORM, n=1, cutoff=0.78)
        if not sugg:
            return df.iloc[0:0].copy()

        best = sugg[0]

        score2 = np.zeros(len(df), dtype=int)
        score2 += 10 * df["title_n"].str.contains(re.escape(best), regex=True).astype(int)
        score2 += 8 * df["aliases_n"].str.contains(re.escape(best), regex=True).astype(int)
        score2 += 3 * df["short_n"].str.contains(re.escape(best), regex=True).astype(int)

        df["score2"] = score2
        hits2 = df[df["score2"] > 0].sort_values(["score2", "title"], ascending=[False, True])
        return hits2.drop(columns=["score2"], errors="ignore").copy()
    
    @reactive.calc
    def candidates():
        df = filtered().copy()
        if len(df) == 0:
            return df
        cols = ["doc_id", "title", "category", "short", "aliases", "url"]
        return df[cols].head(50).copy()

    # -------------------------
    # Spell hint results + apply (based on committed)
    # -------------------------
    @output
    @render.ui
    def spell_hint_results():
        q_raw = (committed_q.get() or "").strip()
        qn = norm_text(q_raw)
        if not qn:
            return ui.HTML("")
        
        df = filtered()

        # toon spell-hint als:
        # - geen resultaten
        # - OF heel weinig resultaten
        show_hint = True

        if not show_hint:
            return ui.HTML("")

        sugg = fuzzy_suggest(qn, VOCAB_NORM, n=1, cutoff=0.72)
        if not sugg or sugg[0] == qn:
            return ui.HTML("")

        pretty = sugg[0]
        return ui.div(
            {"class": "meta"},
            "Bedoelde u misschien: ",
            ui.input_action_link("apply_spell", pretty),
            " ?"
        )

    @reactive.effect
    @reactive.event(input["apply_spell"])
    def _apply_spell():
        q_raw = (committed_q.get() or "").strip()
        qn = norm_text(q_raw)
        sugg = fuzzy_suggest(qn, VOCAB_NORM, n=1, cutoff=0.72)
        if sugg:
            best = sugg[0]
            committed_q.set(best)
            try:
                ui.update_selectize("q", selected=best)
            except Exception:
                pass

            # rerank again
            cand = candidates()
            if cand is None or len(cand) == 0:
                llm_rank.set({"top_ids": [], "did_you_mean": None, "extra_synonyms": [], "reasoning": []})
            else:
                llm_rank.set(llm_rerank(best, cand, k=5))

    # -------------------------
    # LLM hint (did you mean + extra synonyms)
    # -------------------------
    @output
    @render.ui
    def llm_hint():
        q = (committed_q.get() or "").strip()
        if not q:
            return ui.HTML("")
        rank = llm_rank.get()
        did = rank.get("did_you_mean", None)
        syns = rank.get("extra_synonyms", [])

        # fallback: altijd lokale verwante woorden tonen
        if not syns:
            syns = local_related_terms(q, syn, max_n=6)

        parts = []
        if did and did.lower() != q.lower():
            parts.append(
                ui.div(
                    {"class": "meta"},
                    "Bedoelde u: ",
                    ui.input_action_link("apply_llm_spell", did),
                    " ?",
                )
            )
        if syns:
            parts.append(ui.div({"class": "meta"}, f"Verwante woorden: {', '.join(syns)}"))

        return ui.TagList(*parts) if parts else ui.HTML("")

    @reactive.effect
    @reactive.event(input["apply_llm_spell"])
    def _apply_llm_spell():
        rank = llm_rank.get()
        did = rank.get("did_you_mean", None)
        if did:
            committed_q.set(did)
            try:
                ui.update_selectize("q", selected=did)
            except Exception:
                pass

            cand = candidates()
            if cand is None or len(cand) == 0:
                llm_rank.set({"top_ids": [], "did_you_mean": None, "extra_synonyms": [], "reasoning": []})
            else:
                llm_rank.set(llm_rerank(did, cand, k=5))

    # -------------------------
    # Results count + cards
    # -------------------------
    @output
    @render.ui
    def results_count():
        q_raw = (committed_q.get() or "").strip()
        if not q_raw:
            return ui.div({"class": "meta"}, "Typ hierboven een zoekterm en druk op Enter of klik op ‘Zoek’.")
        df = filtered()
        return ui.div({"class": "meta"}, f"Aantal resultaten: {len(df)}")

    @output
    @render.ui
    def results_cards():
        q_raw = (committed_q.get() or "").strip()
        if not q_raw:
            return ui.HTML("")

        df = filtered().head(10).copy()
        if len(df) == 0:
            return ui.div(
                {"class": "no-results"},
                ui.tags.strong("Geen resultaten gevonden."),
                ui.div({"class": "meta"}, "Probeer een ander woord of gebruik AI-navigatie.")
            )

        cards = []
        for _, r in df.iterrows():
            docid = str(r["doc_id"])
            btn_id = f"open_doc_{safe_id(docid)}"

            cards.append(
                ui.div(
                    {"class": "result-card"},
                    ui.div({"class": "result-meta"}, str(r["category"])),

                    ui.div(
                        {"class": "result-title"},
                        ui.tags.a(
                            str(r["title"]),
                            href="#",
                            onclick=f"erfoSetInput('{btn_id}', 1); return false;",
                            style="text-decoration:none; color:inherit;"
                        ),
                    ),

                    ui.div({"class": "result-short"}, str(r["short"]) if nz(r["short"]) else ""),
                    ui.div(
                        {"class": "result-actions"},
                        ui.input_action_button(btn_id, "Lees meer", class_="btn btn-sm btn-primary"),
                    ),
                )
            )

        return ui.TagList(*cards)

    @reactive.effect
    def _wire_open_results_cards():
        df = filtered().head(10).copy()
        for docid in df["doc_id"].astype(str).tolist():
            btn_id = f"open_doc_{safe_id(docid)}"
            if btn_id in input:
                @reactive.effect
                @reactive.event(input[btn_id])
                def _open(docid=docid):
                    selected_id.set(str(docid))
                    current_page.set("detail")

    # -------------------------
    # Detail outputs
    # -------------------------
    @reactive.calc
    def selected_row():
        docid = selected_id.get()
        if not docid:
            return None
        r = content_ui[content_ui["doc_id"].astype(str) == str(docid)]
        return None if len(r) == 0 else r.iloc[0]

    @output
    @render.text
    def detail_title():
        r = selected_row()
        return "" if r is None else r["title"]

    @output
    @render.text
    def detail_meta():
        r = selected_row()
        if r is None:
            return ""
        url = r["url"]
        return f"{r['category']} • {url}" if nz(url) else f"{r['category']}"

    @output
    @render.text
    def detail_short():
        r = selected_row()
        return "" if r is None else r["short"]

    @output
    @render.text
    def detail_long():
        r = selected_row()
        return "" if r is None else r["long"]

    @output
    @render.text
    def detail_next():
        r = selected_row()
        return "" if r is None else (r["next_step"] if nz(r["next_step"]) else "—")

    @output
    @render.text
    def detail_disclaimer():
        r = selected_row()
        return "" if r is None else (r["disclaimer"] if nz(r["disclaimer"]) else "—")

    @output
    @render.ui
    def detail_layer():
        r = selected_row()
        if r is None:
            return ui.HTML("")

        # We tonen altijd de "low" laag (geen hl-knoppen meer)
        docid = r["doc_id"]
        blocks = hl_blocks[hl_blocks["doc_id"].astype(str) == str(docid)].copy()
        bsum = blocks[(blocks["hl_level"] == "low") & (blocks["type"] == "summary")].head(1)
        steps = blocks[(blocks["hl_level"] == "low") & (blocks["type"] == "step")].head(3)
        g = extract_glossary_hits(r["short"] + " " + r["long"], glossary, max_n=6)

        items = [ui.div({"class": "teaser-kicker"}, "Basis – kern & stappen")]
        if len(bsum):
            items.append(ui.p(str(bsum.iloc[0]["content"])))
        if len(steps):
            items.append(ui.tags.ul(*[ui.tags.li(str(x)) for x in steps["content"].tolist()]))

        if len(g):
            items.append(ui.hr())
            items.append(ui.div({"class": "teaser-kicker"}, "Woordenlijst"))
            items.append(
                ui.tags.ul(*[
                    ui.tags.li(ui.tags.strong(str(row["term"])), ": ", str(row["uitleg"]))
                    for _, row in g.iterrows()
                ])
            )
        return ui.card(*items)
    
    @output
    @render.ui
    def home_news_block():
        # Toon alleen als er NOG NIET gezocht is
        q = (committed_q.get() or "").strip()
        if q:
            return ui.HTML("")  # zodra er gezocht is: weg

        return ui.div(
            {"class": "erfo-home-hero"},
            ui.div(
                {"class": "erfo-home-hero-card"},
                ui.div(
                    {"class": "erfo-home-hero-left"},
                    ui.div("ACTUEEL", class_="erfo-home-pill light"),
                    ui.div(
                        "Met trots presenteren we het Meerjarenplan Erfocentrum 2026–2028: Bij de tijd",
                        class_="erfo-home-title-big",
                    ),
                ),
                ui.div(
                    {"class": "erfo-home-hero-right"},
                    ui.div("UITGELICHT", class_="erfo-home-pill ghost"),
                    ui.div(
                        "Zijn kuiltjes in je wangen erfelijk?",
                        class_="erfo-home-title-right",
                    ),
                ),
            ),
        )
    # -------------------------
    # AI modal
    # -------------------------
    def _show_ai_modal():
        m = ui.modal(
            ui.p("Beschrijf je vraag in gewone taal (mag ook in een andere taal)."),
            ui.input_text_area(
                "ai_prompt",
                "Jouw vraag / probleem",
                placeholder="Bijv. 'Is borstkanker erfelijk?' of 'My mother had breast cancer, what about me?'",
                width="100%",
                height="120px",
            ),
            ui.row(
                ui.column(6, ui.input_action_button("ai_send", "Help me stap voor stap", class_="btn-primary")),
                ui.column(6, ui.input_action_button("ai_reset", "Reset gesprek", class_="btn-outline-secondary")),
            ),
            ui.hr(),
            ui.output_ui("ai_chat"),
            ui.hr(),
            ui.output_ui("ai_recs"),
            title="AI-navigatie",
            easy_close=True,
            size="l",
        )
        ui.modal_show(m)

    @reactive.effect
    @reactive.event(input["start_ai_search"])
    def _start_ai_search():
        _show_ai_modal()

    @output
    @render.ui
    def ai_chat():
        h = bot_history.get()
        if not h:
            return ui.div({"class": "meta"}, "Nog geen gesprek. Typ je vraag en klik op ‘Help me stap voor stap’.")
        bubbles = []
        for msg in h:
            role = msg["role"]
            cls = "border rounded p-2 mb-2" if role == "user" else "border rounded p-2 mb-2 bg-light"
            bubbles.append(
                ui.div(
                    {"class": cls},
                    ui.div({"style": "font-weight:600;"}, "Jij" if role == "user" else "Navigator"),
                    ui.div(msg["text"]),
                )
            )
        return ui.TagList(*bubbles)

    @output
    @render.ui
    def ai_recs():
        recs = bot_recs.get()
        if recs is None or len(recs) == 0:
            return ui.div({"class": "meta"}, "Nog geen suggesties.")
        cards = [ui.h4("Top 5 aanbevolen artikelen")]
        for _, r in recs.iterrows():
            docid = str(r["doc_id"])
            cards.append(
                ui.card(
                    ui.div({"class": "result-card"},
                        ui.tags.strong(str(r["title"])),
                        ui.div({"class": "meta"}, f"{r['category']}{(' • ' + r['url']) if nz(r['url']) else ''}"),
                        ui.p(str(r["short"])) if nz(r["short"]) else ui.HTML(""),
                        ui.tags.button(
                            "Open",
                            class_="btn btn-sm btn-primary",
                            onclick=f"erfoSetInput('open_doc', '{docid}')"
                        ),
                    )
                )
            )
        return ui.TagList(*cards)

    @reactive.effect
    @reactive.event(input["open_doc"])
    def _open_doc():
        docid = input["open_doc"]()
        if not nz(docid):
            return
        selected_id.set(str(docid))
        current_page.set("detail")

    @reactive.effect
    @reactive.event(input["ai_send"])
    def _ai_send():
        p_raw = (input["ai_prompt"]() or "").strip()
        if not p_raw:
            return

        mapped = llm_to_dutch_query(p_raw)
        p_nl = mapped["dutch_query"]
        kws = mapped["keywords"]

        combined = " ".join([p_nl] + kws[:8]).strip() or p_raw
        recs = score_docs_for_prompt(combined, content_ui, themes, syn, top_n=5)
        bot_recs.set(recs)

        h = bot_history.get()
        h = h + [{"role": "user", "text": p_raw}]
        if p_nl and p_nl.lower() != p_raw.lower():
            h = h + [{"role": "bot", "text": f"Ik koppel je vraag aan: **{p_nl}**\nTrefwoorden: {', '.join(kws[:8])}"}]

        qn = next_question_for_prompt(p_nl or p_raw)
        h = h + [{"role": "bot", "text": f"Ik toon relevante info op basis van je vraag.\n\nVervolgvraag: {qn}"}]
        bot_history.set(h)

    @reactive.effect
    @reactive.event(input["ai_reset"])
    def _ai_reset():
        bot_history.set([])
        bot_recs.set(content_ui.iloc[0:0].copy())
        ui.update_text_area("ai_prompt", value="")

    @output
    @render.ui
    def ai_help_section():
        q = (committed_q.get() or "").strip()
        if not q:
            return ui.HTML("")

        return ui.div(
            {"class": "erfo-help-card"},
            ui.div(
                {"class": "erfo-help-left"},
                ui.div({"class": "erfo-help-eyebrow"}, "Hulp nodig?"),
                ui.div({"class": "erfo-help-title"}, "Twijfel je waar je moet beginnen?"),
                ui.div(
                    {"class": "erfo-help-text"},
                    "Beschrijf je vraag in gewone taal. We helpen je stap voor stap op weg.",
                ),
            ),
            ui.div(
                {"class": "erfo-help-right"},
                ui.input_action_button(
                    "start_ai_search",
                    "Start AI-navigatie",
                    class_="btn btn-primary btn-erfo-ai",
                ),
            ),
        )

app = App(app_ui, server, static_assets={"": str(APP_DIR / "www")})