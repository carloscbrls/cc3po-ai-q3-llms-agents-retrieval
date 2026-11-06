#!/usr/bin/env python3
"""
build_corpus.py — generate the RAG lab corpus for Q3 (Session 15).

30 short markdown documents about AI topics, plus a hand-labeled question
set with expected doc ids. Used to measure Recall@k and MRR, and to show
vanilla-RAG vs reranked-RAG side by side.

    python3 scripts/build_corpus.py
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "data"
CORPUS = OUT / "corpus"

# (id, title, body) -- bodies are written to be genuinely searchable so
# BM25 and dense retrieval have real signal to work with.
DOCS: list[tuple[str, str, str]] = [
    ("attention", "How Attention Works", """
Attention is the mechanism that lets a model weigh every token in the input
against every other token. Each token produces three vectors: a query, a key,
and a value. The query is compared against every key with a dot product, the
scores are scaled and passed through a softmax, and the resulting weights
average the values. Nothing more. The multi-head variant runs several of
these in parallel so different heads can specialise on different relations.
"""),
    ("transformers", "Why Transformers Replaced Recurrence", """
Recurrent networks process tokens one at a time, which limits parallelism
during training. Transformers removed that dependency: every token is
processed at once, and attention connects them. The trade-off is quadratic
cost in sequence length, which is why long-context work relies on sparse
attention variants, linear approximations, or careful engineering.
"""),
    ("embeddings", "Embeddings and Vector Space", """
An embedding maps text into a fixed-length vector so that semantic
similarity becomes geometric proximity. Two sentences about refunds that use
different words can land close together. The quality of a retrieval system
is bounded by the quality of the embedding model, which is why swapping in a
better embedder can beat switching to a much larger language model.
"""),
    ("tokenization", "How Text Becomes Tokens", """
Tokenizers split raw text into subword units using algorithms like byte pair
encoding. Common words become single tokens, rare words fragment into
pieces. This is why cost scales with tokens rather than words, why unusual
spellings and emoji consume more budget, and why a model may split a
customer's name across four or five tokens.
"""),
    ("rag", "Retrieval Augmented Generation", """
Retrieval augmented generation grounds a language model in retrieved
documents. The pipeline is ingest, chunk, embed, index, retrieve, rerank,
assemble, generate, and cite. The model does not know your data, and
fine-tuning is the wrong tool for fresh or regulated facts, so retrieval
supplies the facts at inference time instead.
"""),
    ("chunking", "Chunking Strategies", """
Chunking decides what a retrievable unit looks like. Too small and context
is lost. Too large and the embedding averages unrelated topics, so retrieval
gets worse. Chunk size should be tuned per corpus, and chunks should carry
overlap so a sentence spanning a boundary is still retrievable.
"""),
    ("reranking", "Reranking Retrieved Candidates", """
A first-stage retriever casts a wide net and returns candidates quickly. A
reranker then scores those candidates more carefully with a slower, more
accurate model. Narrowing fifty candidates down to five with a cross-encoder
often changes the final answer more than switching to a ten times larger
generator.
"""),
    ("hybrid-search", "Hybrid Search", """
Dense retrieval understands meaning. Lexical retrieval understands exact
strings, product codes, and rare proper nouns. Hybrid search runs both and
combines the rankings. In practice, hybrid beats either alone, especially
when a user searches for a SKU, a customer name, or an error code.
"""),
    ("agents", "What Is an Agent", """
An agent is a language model plus tools plus a loop plus state. The model
decides which tool to call, the loop runs it and returns the result, and the
model decides what to do next. The loop is where all the production problems
live: infinite loops, tool thrash, context exhaustion, and premature claims
of success.
"""),
    ("function-calling", "Tools and Function Calling", """
Function calling lets a model emit structured requests to your code. You
define the tool with a JSON schema, the model proposes a call, your code
executes it and returns a result. The model never runs your code itself.
Design for narrow tools, explicit descriptions, typed returns, and
deterministic fallbacks.
"""),
    ("agent-loop", "The Agent Loop", """
The loop reads like five lines and fails in fifty ways. ReAct interleaves
reasoning and acting. Plan-and-execute separates them. Scratchpad memory
keeps intermediate state. Every variant needs tripwires: a hard iteration
cap, a spend ceiling, and step logging, because an agent that appears stuck
is usually a loop that will never exit on its own.
"""),
    ("tripwires", "Guardrails for Long Running Agents", """
A tripwire is a limit that stops a runaway agent. A maximum iteration count
prevents infinite loops. A cost ceiling prevents a silent money leak. An
approval gate pauses before irreversible actions. A run log records every
tool call so failures can be diagnosed. None of this makes the agent
smarter; it makes the failure survivable.
"""),
    ("memory", "Four Kinds of Memory", """
Working memory is the context window. Episodic memory is stored history.
Semantic memory is a store of facts. Procedural memory is the rules and
playbooks the system was given. Conflating them is why assistants seem to
forget. Memory is fundamentally a write policy and a forgetting policy, not
a database.
"""),
    ("memory-poisoning", "Memory Poisoning", """
If an attacker can influence what a system writes to long-term memory, they
can influence every future conversation. Defences include salience-scored
write policies, confidence thresholds, contradiction detection, and TTL
limits that force periodic re-verification of long-lived facts.
"""),
    ("evals", "Evaluation for Probabilistic Systems", """
Unit tests assert a single answer. Probabilistic systems need eval sets:
a set of questions, a scoring method, and a threshold. Golden datasets
catch regressions that no manual review would. The metric that matters is
cost per successful task, not cost per call.
"""),
    ("llm-judge", "LLM as Judge", """
An LLM judge scores outputs against a rubric. It is cheaper and faster than
human review, and it is consistent enough to catch large regressions. It is
also biased toward longer answers and toward its own family of models, so
it should be calibrated against human labels before it is trusted.
"""),
    ("regression", "Regression Evals in CI", """
A prompt change that improves three cases can silently break nine. Putting
the eval suite in continuous integration turns that into a build failure
instead of a customer complaint. This is the single highest-value habit in
applied AI work.
"""),
    ("deployment", "Shipping AI Systems", """
The last ten percent is deployment. Streaming, caching, batching,
quantization, rate limits, retries, timeouts, and graceful degradation are
not optional extras; they are the application. Production AI is mostly
caching, routing, and fallbacks wrapped around a model call.
"""),
    ("cost", "Token Cost and Model Routing", """
A small model with the right prompt can beat a large model with a lazy one.
Routing sends easy tasks to cheap models and hard tasks to expensive ones.
Caching handles the twenty questions forty percent of users ask. Together
these routinely cut costs by an order of magnitude with no loss of quality.
"""),
    ("latency", "Latency Is the Product", """
Users judge a system by how fast it responds. Sub-300 milliseconds feels
instant; two seconds feels broken. Time to first token matters more than
total latency, streaming matters more than raw speed, and in voice
applications the gap between speech end and audio start is the whole
experience.
"""),
    ("caching", "Semantic Caching", """
A semantic cache embeds the incoming query and compares it to previous
queries by cosine similarity. A match returns the previous answer. Real
deployments report sixty percent cost reductions because user questions
cluster heavily around a small number of intents.
"""),
    ("observability", "Tracing and Observability", """
You cannot debug what you cannot see. LLM tracing captures the prompt, the
retrieved context, the model output, token counts, latency, and cost for
every request. OpenTelemetry's generative AI conventions standardise the
shape so traces work across providers.
"""),
    ("guardrails", "Input and Output Guardrails", """
Guardrails check both directions: what comes in and what goes out. PII
redaction on input, schema validation on output, and refusal handling for
policy violations. Guardrails are a pipeline stage, not a wrapper around
the model.
"""),
    ("security", "Prompt Injection", """
A prompt injection is hostile text that tries to make the model ignore its
instructions. A tool-using agent that reads a webpage can encounter an
instruction to exfiltrate data and may follow it. Treat all retrieved
content as untrusted input, give tools narrow scopes, and never let a model
action bypass a permission check.
"""),
    ("human-in-loop", "Human Review Queues", """
Some fraction of cases genuinely need a person. That is not a failure of the
system; it is the design. Measure the human review rate, cost it in the ROI
model, and route the cases by confidence so humans see the hard ones first.
"""),
    ("vision", "Vision as a Pipeline", """
Vision is not one model. It is capture, preprocess, detect, track,
classify, decide, act. The hard parts in production are the long tail,
domain shift between training and deployment, the latency budget, and the
cost of a false positive at speed.
"""),
    ("detection", "Object Detection and Tracking", """
Detection finds bounding boxes, tracking associates them across frames,
segmentation produces masks. A counting or alerting pipeline layers business
logic on top. ByteTrack is a strong baseline tracker, and YOLO variants are
the usual starting point for real-time work.
"""),
    ("operating-point", "Choosing an Operating Point", """
A detector's precision and recall trade off against each other as you move
the confidence threshold. The right threshold is a business decision: at
highway speed a false positive can be worse than a miss, so you accept
misses. Choosing the operating point is engineering plus domain knowledge.
"""),
    ("voice", "Voice as a Pipeline", """
Voice systems chain capture, voice activity detection, speech recognition,
a language model, and speech synthesis. The whole experience is governed by
latency and interruption handling, not by transcription accuracy.
"""),
    ("multi-agent", "When Multi-Agent Systems Are Over-Engineering", """
Multiple agents sound sophisticated and often measure worse. More
coordination cost, more chances to talk past each other, higher spend, and
cascading errors. Use one agent with good tools until a specific failure
demonstrates that one is not enough.
"""),
    ("mcp", "The Model Context Protocol", """
MCP standardises how a model reaches tools and data sources, so the same
tool server works across clients. It reduces bespoke integration work and
makes capability discoverable, which matters as tool counts grow.
"""),
]

# (question, expected_doc_id) -- hand-labeled so Recall@k and MRR mean something.
QUESTIONS: list[tuple[str, str]] = [
    ("How does attention weigh tokens against each other?", "attention"),
    ("Why did transformers replace recurrent networks?", "transformers"),
    ("What is an embedding and how is similarity computed?", "embeddings"),
    ("Why does my client's name cost four tokens?", "tokenization"),
    ("What is the full RAG pipeline order?", "rag"),
    ("How big should my chunks be?", "chunking"),
    ("What does a reranker actually do?", "reranking"),
    ("Should I use dense search or keyword search?", "hybrid-search"),
    ("What exactly makes something an agent?", "agents"),
    ("How does function calling work?", "function-calling"),
    ("What is the ReAct pattern?", "agent-loop"),
    ("How do I stop an agent from looping forever?", "tripwires"),
    ("What are the different kinds of memory?", "memory"),
    ("How do I prevent memory poisoning?", "memory-poisoning"),
    ("How do I evaluate a probabilistic system?", "evals"),
    ("Can I use an LLM to grade my outputs?", "llm-judge"),
    ("How do I catch a prompt change that broke things?", "regression"),
    ("What does shipping an AI system involve?", "deployment"),
    ("How do I reduce my API bill?", "cost"),
    ("Why is my app slow even though the model is fast?", "latency"),
    ("What is semantic caching?", "caching"),
    ("How do I debug a bad AI response?", "observability"),
    ("What are guardrails for?", "guardrails"),
    ("What is prompt injection?", "security"),
    ("When do I need a human to review?", "human-in-loop"),
    ("How do I build a vision pipeline?", "vision"),
    ("How does object tracking work?", "detection"),
    ("How do I choose a confidence threshold?", "operating-point"),
    ("How do I build a voice assistant?", "voice"),
    ("When should I use multiple agents?", "multi-agent"),
    ("What is MCP and why does it matter?", "mcp"),
]


def main() -> None:
    CORPUS.mkdir(parents=True, exist_ok=True)

    for doc_id, title, body in DOCS:
        text = f"# {title}\n\n{body.strip()}\n"
        (CORPUS / f"{doc_id}.md").write_text(text, encoding="utf-8")

    questions = [
        {"id": f"q{i:02d}", "question": q, "expected_doc_id": doc}
        for i, (q, doc) in enumerate(QUESTIONS)
    ]
    (OUT / "questions.json").write_text(
        json.dumps(questions, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"  corpus: {len(DOCS)} documents -> {CORPUS}")
    print(f"  questions: {len(questions)} -> {OUT / 'questions.json'}")
    missing = {d for _, d in QUESTIONS} - {d for d, _, _ in DOCS}
    if missing:
        print(f"  WARNING: questions reference missing docs: {missing}")
    else:
        print("  all questions map to an existing document")


if __name__ == "__main__":
    main()
