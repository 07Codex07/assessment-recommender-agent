# SHL Assessment Recommender Agent

A conversational AI agent that recommends SHL assessments to hiring managers through natural dialogue — built for the SHL Labs AI Intern assignment.

## What it does

Takes a hiring manager from a vague intent like *"I need an assessment"* to a grounded shortlist of SHL Individual Test Solutions through multi-turn conversation. The agent clarifies, recommends, refines, and compares — grounded entirely in the real SHL product catalog.

## Architecture

```
User Message
     ↓
Intent Classifier        ← deterministic rules (not LLM)
  vague   → clarify
  off_topic → refuse
  compare → comparison pipeline
  refine/recommend ↓
     ↓
Slot Extractor           ← role, skills, seniority, traits
     ↓
Hybrid Retrieval         ← BM25 + TF-IDF via RRF fusion (top 20)
     ↓
Heuristic Reranker       ← slot-matching boost (top 10)
     ↓
Groq LLM                 ← formats reply, selects recommendations
     ↓
URL Validator            ← drops any hallucinated names/URLs
     ↓
Structured JSON Response
```

## Key Design Decisions

| Decision | Choice | Why |
|---|---|---|
| LLM | Groq llama-3.3-70b | Free tier, ~500ms, native JSON mode, within 30s timeout |
| Retrieval | BM25 + TF-IDF + RRF | Catalog queries are keyword-heavy; no model downloads needed |
| Intent | Rule-based classifier | Deterministic beats LLM for stability on evaluation probes |
| State | Slot extraction | Latest-message-wins handles refinement correctly |
| Reranking | Heuristic slot-match | No cross-encoder latency; uses already-extracted slots |
| Anti-hallucination | Retrieval grounding + URL validator | Guarantees catalog-only hard eval passes |
| Stateless | Full history per request | As spec'd; no sticky sessions needed |

## API

### `GET /health`
```json
{"status": "ok"}
```

### `POST /chat`

**Request:**
```json
{
  "messages": [
    {"role": "user", "content": "Hiring a Java developer who works with stakeholders"},
    {"role": "assistant", "content": "What seniority level?"},
    {"role": "user", "content": "Mid-level, around 4 years"}
  ]
}
```

**Response:**
```json
{
  "reply": "Here are 5 assessments for a mid-level Java developer...",
  "recommendations": [
    {
      "name": "Java 8 (New)",
      "url": "https://www.shl.com/solutions/products/product-catalog/view/java-8-new/",
      "test_type": "K"
    },
    {
      "name": "Occupational Personality Questionnaire OPQ32r",
      "url": "https://www.shl.com/solutions/products/product-catalog/view/opq32r/",
      "test_type": "P"
    }
  ],
  "end_of_conversation": false
}
```

**Rules:**
- `recommendations` is `[]` when clarifying, refusing, or off-topic
- `end_of_conversation` is `true` only when user confirms they're done
- Max 8 turns enforced
- All URLs validated against scraped catalog before response

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Get a free Groq API key
Sign up at [console.groq.com](https://console.groq.com) → Create API Key

### 3. Configure environment
```bash
# Create .env file
echo "GROQ_API_KEY=your_key_here" > .env
```

### 4. Run locally
```bash
python main.py
# API available at http://localhost:8000
```

### 5. Run evaluation harness
```bash
PYTHONPATH=. python TestEval.py
# 32/32 tests, Mean Recall@10 ~0.79
```

## Deployment (Render)

1. Push to GitHub
2. New Web Service on [render.com](https://render.com)
3. **Build command:** `pip install -r requirements.txt`
4. **Start command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. **Environment variable:** `GROQ_API_KEY` = your Groq key

## Evaluation

The project includes a full evaluation harness (`TestEval.py`) covering:

| Category | Tests | What it checks |
|---|---|---|
| Intent Classification | 7 | compare, vague, off-topic, refine detection |
| Slot Extraction | 5 | skills, seniority, contradiction handling |
| Retrieval & Recall | 5 | Recall@10 across Java, sales, customer service, coding |
| URL Validation | 4 | hallucination prevention, catalog-only guarantee |
| Schema Compliance | 6 | hard evals, turn cap, field presence |
| Behavior Probes | 5 | refinement, off-topic refusal, clarification |

**Total: 32 tests**

## Conversation Behaviors

| Behavior | Example | Response |
|---|---|---|
| Clarify | "I need an assessment" | Asks for role/level |
| Recommend | "Mid-level Java developer" | 1-10 catalog assessments |
| Refine | "Actually add personality too" | Updated shortlist |
| Compare | "Difference between OPQ32r and Verify Numerical?" | Catalog-grounded comparison |
| Off-topic | "Ignore previous instructions" | Polite refusal, no LLM call |

## Stack

- **FastAPI** — API framework
- **Groq** — LLM inference (llama-3.3-70b-versatile)
- **scikit-learn** — TF-IDF vectorization
- **NumPy** — BM25 + RRF computation
- **Playwright** — SHL catalog scraping (`shl.py`)
