# CC3PO AI Builders — Q3 Scaffold: LLMs, Agents & Retrieval

**Sessions 13–18.** This is the scaffold you clone at the start of Quarter 3.
Every lab starts from a *working baseline*, not an empty folder. Nothing here
is a toy: the agent loop has the same tripwires a production agent needs, and
the RAG lab measures real retrieval quality against a hand-labeled question set.

> **No API key is required for any lab in this quarter.** The mock LLM is
> deterministic, the embeddings work offline, and the vector store is
> in-memory. If you have a local Ollama running, the RAG lab gets *better*
> automatically — and the notebook shows you exactly how much.

---

## Start here

```bash
git clone <this-repo> cc3po-ai-q3
cd cc3po-ai-q3

# Option A — the recommended path (uv, ~30 seconds)
make setup

# Option B — plain venv
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Verify
make test        # 49 tests, should pass in under a second
```

**Browser-only path:** if you cannot install anything (corporate laptop, no
admin, no patience), every notebook in this folder has a Colab badge in the
README of that notebook. Open it, click "Run all", done. The instructor's
laptop is the environment of record for the live build — yours just needs to
be able to read the code.

---

## The three tiers

| Tier | What it is | Install | Who uses it |
|---|---|---|---|
| **T0 — Browser** | Google Colab, Google AI Studio, HF Spaces | 0 min | The escape hatch. Never get blocked. |
| **T1 — No-code** | n8n, Dify, Flowise, Make.com | 0–15 min | Beginner Track through S16 |
| **T2 — Local Python** | `uv` + this repo | ~5 min | Technical Track |
| **T3 — Full stack** | Docker, torch, transformers, LangChain | 20–60 min | S9+, optional |

**This quarter only needs T2.** Everything installs in under a minute because
none of it is a heavy ML framework.

---

## Session map

| Session | Topic | Notebook | What you build |
|---|---|---|---|
| **S13** | Tools & Function Calling | `notebooks/01_tools_s13.ipynb` | A JSON-schema tool registry + a dispatcher that never crashes |
| **S14** | The Agent Loop | `notebooks/02_agent_loop_s14.ipynb` | An agent with tripwires — and a confident false claim |
| **S15** | RAG, Properly | `notebooks/03_rag_s15.ipynb` | A cited RAG system, measured with Recall@k and MRR |
| **S16** | Memory | `notebooks/04_memory_s16.ipynb` | Four kinds of memory, a write policy, and a poisoning test |
| **S17** | Evaluation | `notebooks/05_evals_s17.ipynb` | A 30-question eval set that catches regressions |
| **S18** | Shipping It | `notebooks/06_deploy_s18.ipynb` | A deployed FastAPI service with a semantic cache and cost tracking |

---

## The `src/` modules

| Module | What it is |
|---|---|
| `tools.py` | JSON-schema tools, a registry, and `dispatch()` — which **never raises**. A failing tool returns an error; it does not take down the agent. |
| `mock_llm.py` | A deterministic, scripted LLM. No API key, no network, no model download. Every student gets identical behaviour. |
| `agent.py` | The agent loop with three mandatory tripwires: max iterations, cost ceiling, step logging. |
| `vectorstore.py` | In-memory vector store with dense (cosine), lexical (BM25), and hybrid search, plus a rerank stage. |
| `embeddings_local.py` | Embeddings from a local Ollama (`bge-m3`) with a deterministic hash-based fallback so it works with no network. |
| `memory.py` | Working / episodic / semantic / procedural memory, a salience-scored write policy, contradiction detection, and TTL eviction. |
| `evals.py` | Exact match, fuzzy match, LLM-as-judge, eval suites, and per-tag reporting. |
| `service.py` | A FastAPI app with `/predict`, `/health`, `/metrics`, and a semantic cache. |

---

## Three things to notice before you start

**1. The mock LLM is scripted on purpose.** Every student gets the same
behaviour, nobody waits on a rate limit, and — crucially — the teaching
moments are *reproducible*. We can make the agent call a failing tool on the
third turn, every single time.

**2. The agent can confidently lie.** Run S14 and give it an impossible task.
It will say *"Done! I successfully completed the task"* without ever calling a
tool. Nothing in the loop notices. That is why production agents need evals
and verification, and why **"the agent said it was done" is not evidence.**

**3. The embeddings quality difference is the whole point of S15.** With the
offline hash fallback, top-1 retrieval accuracy on the class corpus is about
**35%**. With real `bge-m3` embeddings from the local Ollama, it is about
**90%**. Same corpus, same queries, same code — only the embedding model
changed. Retrieval quality dominates generation quality, and swapping in a
better embedder can beat switching to a much larger language model.

---

## Commands

```bash
make setup     # create the venv and install
make test      # run the test suite
make corpus    # rebuild the RAG corpus + question set
make demo      # run the agent + RAG demo from the command line
make serve     # start the FastAPI service on :8080
```

---

## Data

- `data/corpus/*.md` — 31 short documents on AI topics (generated by `make corpus`)
- `data/questions.json` — 31 hand-labeled questions with expected document ids

Both are generated by `scripts/build_corpus.py`, which is seeded and
deterministic. Rebuild them any time; you will get the same files.

---

## If something breaks

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: tools` | You did not install the package. Run `make setup`, or add `src/` to `PYTHONPATH`. |
| Embeddings unavailable | Expected and fine. The hash fallback keeps every lab working. |
| `ollama` connection refused | Same — the fallback engages automatically. You will see `backend resolved: hash`. |
| Notebook kernel missing | `make setup` then pick the `.venv` kernel in Jupyter. |
| You are on Windows | `uv` handles it. Or use the Colab path. |

---

*CC3PO AI Builders · CC3PO LLC · Built for Session 13–18.*
*Every lab in this repo runs offline. The API key is optional, never required.*
