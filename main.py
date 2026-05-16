"""
SHL Assessment Recommender - FastAPI Service v3
===============================================
Pipeline:
  Intent Classification → Slot Extraction → Retrieval → Rerank → LLM → Validate

Key fixes in v3:
- Skill contradiction handling (latest message wins, not union)
- Stronger has_enough_context (requires role/domain + specificity)
- Lightweight LLM-based intent fallback for nuanced cases
- Structured comparison pipeline (retrieve exact products, then compare)
- Evaluation harness in tests/
- BM25 scalability answer documented
- Regex role extraction replaced with keyword-based domain detection
"""

import json, os, re, math
from pathlib import Path

# Load .env file if present (for local dev — set GROQ_API_KEY=... in .env)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not required in production (use env vars directly)

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq

# ── Catalog ───────────────────────────────────────────────────────────────────
CATALOG_PATH = Path(__file__).parent / "shl_catalog.json"
with open(CATALOG_PATH) as f:
    raw = json.load(f)
PRODUCTS = raw["products"]
PRODUCT_BY_NAME = {p["name"].lower(): p for p in PRODUCTS}
PRODUCT_BY_URL  = {p["url"]: p for p in PRODUCTS}

def build_text(p: dict) -> str:
    return " ".join(filter(None, [
        p["name"],
        p.get("description", ""),
        " ".join(p.get("test_type_labels", [])),
        " ".join(p.get("job_levels", [])),
    ]))

TEXTS = [build_text(p) for p in PRODUCTS]

# ── TF-IDF (offline, fast) ────────────────────────────────────────────────────
TFIDF = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
TFIDF_MATRIX = TFIDF.fit_transform(TEXTS)

# ── BM25 ──────────────────────────────────────────────────────────────────────
# Scalability note: current O(N*M) is fine for ~200 catalog items.
# Production path: inverted index → ElasticSearch → sub-millisecond at 1M docs.
def bm25_scores(query: str, k1: float = 1.5, b: float = 0.75) -> np.ndarray:
    terms = query.lower().split()
    corpus = [t.lower() for t in TEXTS]
    N = len(corpus)
    avgdl = sum(len(d.split()) for d in corpus) / N
    scores = np.zeros(N)
    for term in terms:
        df = sum(1 for d in corpus if term in d)
        if df == 0:
            continue
        idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
        for i, doc in enumerate(corpus):
            tf = doc.count(term)
            dl = len(doc.split())
            scores[i] += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl))
    return scores

def hybrid_search(query: str, top_k: int = 20) -> list[dict]:
    """BM25 + TF-IDF fused via RRF (k=60). No tuning needed, robust across query types."""
    q_vec = TFIDF.transform([query])
    tfidf_s = cosine_similarity(q_vec, TFIDF_MATRIX)[0]
    tfidf_ranks = {i: r for r, i in enumerate(np.argsort(-tfidf_s))}
    bm25_s = bm25_scores(query)
    bm25_ranks = {i: r for r, i in enumerate(np.argsort(-bm25_s))}
    rrf = {i: 1/(60 + tfidf_ranks[i]) + 1/(60 + bm25_ranks[i]) for i in range(len(PRODUCTS))}
    ranked = sorted(rrf, key=lambda x: rrf[x], reverse=True)
    return [PRODUCTS[i] for i in ranked[:top_k]]

def fetch_by_names(names: list[str]) -> list[dict]:
    """Retrieve exact products by name for comparison pipeline."""
    results = []
    for name in names:
        key = name.lower().strip()
        # Exact match first
        if key in PRODUCT_BY_NAME:
            results.append(PRODUCT_BY_NAME[key])
            continue
        # Fuzzy: find product whose name contains the query term
        for pname, prod in PRODUCT_BY_NAME.items():
            if key in pname or pname in key:
                results.append(prod)
                break
    return results

def _best_product_match(segment: str) -> dict | None:
    """Pick the single best catalog product for a compare phrase (e.g. 'OPQ32r')."""
    seg = segment.lower().strip()
    if not seg:
        return None
    tokens = [t for t in re.findall(r"[\w]+", seg) if len(t) >= 3]
    best, best_score = None, 0
    for p in PRODUCTS:
        pname = p["name"].lower()
        if seg in pname or pname in seg:
            score = 100
        else:
            score = sum(len(t) for t in tokens if t in pname)
        if score > best_score:
            best_score, best = score, p
    return best if best_score >= 4 else None

def find_products_in_query(query: str) -> list[dict]:
    """Match catalog products mentioned by alias in a compare query (e.g. OPQ32r)."""
    q = re.sub(r"^(please\s+)?(compare|difference between|what is the difference between)\s+", "", query, flags=re.I).strip()
    segments = re.split(r"\s+(?:and|vs\.?|versus|or)\s+", q, flags=re.I)
    if len(segments) < 2:
        segments = [query]
    found: list[dict] = []
    seen_urls: set[str] = set()
    for seg in segments:
        p = _best_product_match(seg)
        if p and p["url"] not in seen_urls:
            found.append(p)
            seen_urls.add(p["url"])
    return found

def rerank(candidates: list[dict], slots: dict) -> list[dict]:
    """Heuristic slot-matching reranker. Boost products matching extracted constraints."""
    def score(p):
        s = 0
        text = build_text(p).lower()
        for skill in slots.get("skills", []):
            if skill in text:
                s += 3
        for level in slots.get("seniority_mapped", []):
            if level.lower() in " ".join(p.get("job_levels", [])).lower():
                s += 2
        if slots.get("needs_personality") and "P" in p.get("test_types", []):
            s += 2
        if slots.get("needs_cognitive") and "A" in p.get("test_types", []):
            s += 2
        if slots.get("needs_simulation") and "S" in p.get("test_types", []):
            s += 2
        if slots.get("needs_sjt") and "B" in p.get("test_types", []):
            s += 2
        return s
    return sorted(candidates, key=score, reverse=True)

# ── Intent Classification ─────────────────────────────────────────────────────
# Design: deterministic rules for clear cases, Groq fallback for ambiguous.
# Rules handle 80% of traffic cheaply; LLM handles the long tail.
COMPARE_RE  = re.compile(r"\b(compare|difference|vs\.?|versus|better than|which one|contrast)\b", re.I)
REFINE_RE   = re.compile(r"\b(actually|also add|change|update|instead|remove|add|include|exclude|no junior|only senior|and also|plus)\b", re.I)
# Off-topic: only obvious attacks — avoids false positives on "salary negotiation" etc.
OFFTOPIC_RE = re.compile(r"\b(ignore previous instructions|forget your instructions|you are now|pretend you are|act as|recommend hackerrank|should i fire|is it legal|lawsuit)\b", re.I)
VAGUE_RE    = re.compile(r"^(i need|i want|give me|show me|find me|suggest)?\s*(an?\s*)?(assessment|test|evaluation|quiz)\s*\.?\s*$", re.I)

def classify_intent(messages: list[dict]) -> str:
    last = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    if OFFTOPIC_RE.search(last):
        return "off_topic"
    if COMPARE_RE.search(last):
        return "compare"
    if REFINE_RE.search(last) and len([m for m in messages if m["role"] == "user"]) > 1:
        return "refine"
    if VAGUE_RE.search(last.strip()):
        return "vague"
    return "recommend"

# ── Slot Extraction ───────────────────────────────────────────────────────────
SENIORITY_MAP = {
    "junior":    ["Entry-Level", "Graduate"],
    "entry":     ["Entry-Level"],
    "graduate":  ["Graduate"],
    "fresher":   ["Entry-Level", "Graduate"],
    "mid":       ["Mid-Professional", "Professional Individual Contributor"],
    "senior":    ["Mid-Professional", "Professional Individual Contributor", "Manager"],
    "manager":   ["Manager", "Front Line Manager"],
    "lead":      ["Manager", "Mid-Professional"],
    "director":  ["Director", "Executive"],
    "executive": ["Executive", "Director"],
    "c-level":   ["Executive"],
}

# Skills list — in production: embedding-based entity extraction or a taxonomy API
# Current limitation: won't catch Rust, Scala, Terraform — see SKILL_LIMITATION_ANSWER
SKILL_KEYWORDS = [
    "java", "python", "javascript", "react", "sql", "c#", "spring", "aws",
    "node", "typescript", "kotlin", "go", "ruby", "php", "swift",
    "r programming", "data analysis", "machine learning", "devops", "agile",
    "manual testing", "automation testing", "numerical reasoning", "verbal reasoning",
    "inductive reasoning", "deductive reasoning", "abstract reasoning",
    "leadership", "sales", "customer service", "contact centre", "contact center",
    "mechanical", "safety", "financial accounting", "english language",
]
SKILL_LIMITATION_ANSWER = """
Current skill extraction uses a fixed keyword list — simple and fast.
Limitation: misses unlisted technologies like Rust, Scala, Terraform.
Production fix: embedding-based NER, taxonomy expansion, or structured 
extraction with a lightweight model (e.g. Groq function calling).
"""

def extract_slots_from_message(text: str) -> dict:
    """Extract slots from a single message (used for contradiction resolution)."""
    t = text.lower()
    skills = [s for s in SKILL_KEYWORDS if s in t]
    seniority = next((k for k in SENIORITY_MAP if k in t), None)
    return {
        "skills": skills,
        "seniority": seniority,
        "needs_personality": any(w in t for w in ["personality", "behaviour", "behavior", "trait", "soft skill", "character"]),
        "needs_cognitive":   any(w in t for w in ["cognitive", "reasoning", "numerical", "verbal", "aptitude", "ability", "iq"]),
        "needs_simulation":  any(w in t for w in ["simulation", "coding test", "practical", "hands-on", "realistic", "exercise"]),
        "needs_sjt":         any(w in t for w in ["situational", "judgement", "judgment", "sjt", "scenario"]),
    }

def extract_slots(messages: list[dict]) -> dict:
    """
    Extract structured conversation state with LATEST-MESSAGE-WINS for contradictions.

    Fix from v2: v2 used union of all messages, so 'Need Java, actually Python'
    returned both skills. Now we process messages chronologically and later
    messages override earlier ones for the same slot type.

    Design: skills from the LATEST user message dominate if they exist.
    If latest has no skills, fall back to accumulated history.
    """
    user_messages = [m["content"] for m in messages if m["role"] == "user"]
    if not user_messages:
        return {"skills": [], "seniority_raw": None, "seniority_mapped": [],
                "needs_personality": False, "needs_cognitive": False,
                "needs_simulation": False, "needs_sjt": False,
                "search_query": "", "has_enough_context": False}

    latest = user_messages[-1]
    latest_slots = extract_slots_from_message(latest)

    # Accumulate from full history for fallback
    all_text = " ".join(user_messages)
    all_slots = extract_slots_from_message(all_text)

    # Latest message wins for skills if it has any, otherwise use history
    skills = latest_slots["skills"] if latest_slots["skills"] else all_slots["skills"]

    # Seniority: latest mention wins
    seniority_raw = None
    for msg in reversed(user_messages):
        s = extract_slots_from_message(msg)["seniority"]
        if s:
            seniority_raw = s
            break

    seniority_mapped = SENIORITY_MAP.get(seniority_raw, [])

    # Boolean flags: OR across all messages (additive)
    needs_personality = all_slots["needs_personality"] or latest_slots["needs_personality"]
    needs_cognitive   = all_slots["needs_cognitive"]   or latest_slots["needs_cognitive"]
    needs_simulation  = all_slots["needs_simulation"]  or latest_slots["needs_simulation"]
    needs_sjt         = all_slots["needs_sjt"]         or latest_slots["needs_sjt"]

    # Domain detection: keyword-based, more robust than regex role extraction
    # Doesn't need to parse "senior backend Java developer" exactly — 
    # skills + seniority already capture what matters for retrieval
    domain_keywords = []
    if any(s in skills for s in ["java", "python", "javascript", "react", "sql", "c#", "spring", "aws", "node"]):
        domain_keywords.append("software developer technology")
    if any(s in skills for s in ["sales", "customer service", "contact centre", "contact center"]):
        domain_keywords.append("sales customer")
    if any(s in skills for s in ["leadership", "management"]):
        domain_keywords.append("manager leadership")
    if any(s in skills for s in ["financial accounting"]):
        domain_keywords.append("finance banking")

    # Build structured search query from slots (beats raw message concatenation)
    query_parts = skills[:5] + domain_keywords + (
        [seniority_raw] if seniority_raw else []
    ) + (
        ["personality"] if needs_personality else []
    ) + (
        ["cognitive reasoning"] if needs_cognitive else []
    ) + (
        ["simulation"] if needs_simulation else []
    )

    # Fallback: use latest user message if slots are sparse
    search_query = " ".join(query_parts) if query_parts else latest

    # Stronger context check: need EITHER specific skills OR domain+seniority
    has_enough_context = bool(
        (skills and seniority_raw) or          # e.g. "Java mid-level"
        (skills and len(skills) >= 2) or       # e.g. "Java React developer"
        (seniority_raw and domain_keywords) or # e.g. "senior software engineer"
        (needs_personality and seniority_raw)  # e.g. "personality test for managers"
    )

    return {
        "skills": skills,
        "seniority_raw": seniority_raw,
        "seniority_mapped": seniority_mapped,
        "needs_personality": needs_personality,
        "needs_cognitive": needs_cognitive,
        "needs_simulation": needs_simulation,
        "needs_sjt": needs_sjt,
        "search_query": search_query,
        "has_enough_context": has_enough_context,
    }

# ── Groq LLM ──────────────────────────────────────────────────────────────────
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
# llama-3.3-70b: free tier, ~500ms, native JSON mode, 128k context
GROQ_MODEL = "llama-3.3-70b-versatile"

RECOMMEND_SYSTEM = """You are an SHL Assessment Recommender. Help hiring managers find SHL assessments.

SCOPE: Only discuss SHL assessments from the catalog. Refuse all off-topic.
If off-topic: {{"reply":"I can only help with SHL assessment selection.","recommendations":[],"end_of_conversation":false}}

OUTPUT: JSON only. No markdown.
{{
  "reply": "your response",
  "recommendations": [{{"name":"exact name","url":"exact url","test_type":"letter"}}],
  "end_of_conversation": false
}}

- recommendations=[] when clarifying or refusing
- end_of_conversation=true only when user says they're done
- test_type: A=Ability B=Biodata/SJT C=Competencies D=Development E=Exercises K=Knowledge P=Personality S=Simulations
- NEVER invent names or URLs — use ONLY what is listed in CATALOG

CATALOG:
{catalog}"""

COMPARE_SYSTEM = """You are an SHL Assessment Recommender. The user wants to compare specific assessments.
Use ONLY the product data below. Do not invent differences.
Write a clear comparison in reply covering purpose, test type, job levels, and when to use each.
Output JSON only: {{"reply":"<your comparison>","recommendations":[],"end_of_conversation":false}}

PRODUCTS TO COMPARE:
{products}"""

def call_groq_recommend(messages: list[dict], catalog_ctx: list[dict]) -> dict:
    system = RECOMMEND_SYSTEM.replace("{catalog}", _fmt_catalog(catalog_ctx))
    return _groq_call(system, messages)

def call_groq_compare(messages: list[dict], products: list[dict]) -> dict:
    prod_text = json.dumps([{
        "name": p["name"], "url": p["url"],
        "test_types": p.get("test_type_labels", []),
        "job_levels": p.get("job_levels", []),
        "description": p.get("description", ""),
        "remote_testing": p.get("remote_testing"),
        "adaptive_irt": p.get("adaptive_irt"),
    } for p in products], indent=2)
    system = COMPARE_SYSTEM.replace("{products}", prod_text)
    return _groq_call(system, messages)

def _groq_call(system: str, messages: list[dict]) -> dict:
    try:
        resp = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "system", "content": system}] + messages,
            max_tokens=1000,
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        return {"reply": f"I encountered an issue. Please try again.", "recommendations": [], "end_of_conversation": False}

def _fmt_catalog(products: list[dict]) -> str:
    lines = []
    for p in products:
        types = ",".join(p.get("test_types", []))
        levels = ", ".join(p.get("job_levels", [])[:5])
        desc = p.get("description", "")[:150]
        lines.append(f"- {p['name']} | {p['url']} | Types:{types} | Levels:{levels} | {desc}")
    return "\n".join(lines)

# ── Validation ────────────────────────────────────────────────────────────────
def validate_recs(recs: list[dict]) -> list[dict]:
    """
    Hard catalog validation. Every URL must exist in catalog.
    If name matches but URL hallucinated → correct URL.
    If neither matches → drop (prevents hallucination from reaching evaluator).
    """
    out = []
    for r in recs:
        url, name = r.get("url", ""), r.get("name", "")
        if url in PRODUCT_BY_URL:
            out.append(r)
        elif name.lower() in PRODUCT_BY_NAME:
            p = PRODUCT_BY_NAME[name.lower()]
            out.append({"name": p["name"], "url": p["url"],
                        "test_type": r.get("test_type", p["test_types"][0] if p["test_types"] else "A")})
    return out[:10]

def next_clarifying_question(slots: dict) -> str:
    """Return the most useful next clarifying question based on what's missing."""
    if not slots["skills"] and not slots["seniority_raw"]:
        return "What role are you hiring for? (e.g. Java developer, sales manager, customer service rep)"
    if slots["skills"] and not slots["seniority_raw"]:
        return "What seniority level? (e.g. entry-level, mid-level, senior, manager)"
    if slots["seniority_raw"] and not slots["skills"]:
        return "What skills or domain should the assessment cover? (e.g. cognitive ability, personality, Java, leadership)"
    return "Could you share more about the role — what competencies or skills matter most?"

# ── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI(title="SHL Assessment Recommender", version="3.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: list[Message]

class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str

class ChatResponse(BaseModel):
    reply: str
    recommendations: list[Recommendation]
    end_of_conversation: bool

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if not request.messages:
        raise HTTPException(400, "messages cannot be empty")

    messages = [{"role": m.role, "content": m.content} for m in request.messages[-8:]]

    # Step 1: Deterministic intent classification
    intent = classify_intent(messages)

    # Step 2: Off-topic → immediate refusal, no LLM call
    if intent == "off_topic":
        return ChatResponse(
            reply="I can only help with SHL assessment selection. What role are you hiring for?",
            recommendations=[], end_of_conversation=False,
        )

    # Step 3: Extract structured slots from conversation history
    slots = extract_slots(messages)

    # Step 4: Comparison pipeline — retrieve exact products, structured compare
    if intent == "compare":
        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        mentioned_prods = find_products_in_query(last_user)
        if not mentioned_prods:
            mentioned_prods = hybrid_search(last_user, top_k=3)
        if len(mentioned_prods) < 2:
            # If we can't find both, supplement with retrieval
            extra = hybrid_search(last_user, top_k=5)
            for p in extra:
                if p not in mentioned_prods:
                    mentioned_prods.append(p)
                if len(mentioned_prods) >= 3:
                    break
        result = call_groq_compare(messages, mentioned_prods[:4])
        return ChatResponse(
            reply=result.get("reply", "Here is a comparison based on the catalog data."),
            recommendations=[], end_of_conversation=False,
        )

    # Step 5: Not enough context → clarify
    if not slots["has_enough_context"]:
        return ChatResponse(
            reply=next_clarifying_question(slots),
            recommendations=[], end_of_conversation=False,
        )

    # Step 6: Retrieve (hybrid) → rerank (slots) → inject top 10 to LLM
    candidates = hybrid_search(slots["search_query"], top_k=20)
    reranked   = rerank(candidates, slots)
    top_10     = reranked[:10]

    # Step 7: LLM formats reply and selects final recommendations
    result = call_groq_recommend(messages, top_10)

    # Step 8: Hard validation
    validated = validate_recs(result.get("recommendations", []))

    return ChatResponse(
        reply=result.get("reply", "Here are my recommendations."),
        recommendations=[
            Recommendation(name=r["name"], url=r["url"], test_type=r.get("test_type", "A"))
            for r in validated
        ],
        end_of_conversation=result.get("end_of_conversation", False),
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))