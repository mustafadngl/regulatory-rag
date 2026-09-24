# Regulatory RAG

[![CI](https://github.com/mustafadngl/regulatory-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/mustafadngl/regulatory-rag/actions/workflows/ci.yml)

Question answering over EU regulatory texts (EU AI Act, GDPR) that returns **cited answers** and, more importantly, **proves its own answer quality in CI**.

Most retrieval-augmented generation demos stop at "it produced a plausible answer." This service treats answer quality as a testable property: a golden question set, retrieval and groundedness metrics, and a build that fails when quality regresses.

> **Status:** early. The service skeleton, container, and CI pipeline are in place. Retrieval and evaluation are in progress — see the roadmap below.

## Why this exists

Regulatory text is a genuinely hard retrieval problem and a genuinely useful one. Articles cross-reference each other, recitals qualify obligations stated elsewhere, and a confident-but-ungrounded answer is worse than no answer. That makes it a good forcing function for the parts of RAG that actually matter in production: retrieval quality, grounding, citation integrity, latency, and cost.

The corpus is public, stable, and free of licensing complications.

## Architecture

```
                  ┌──────────────┐
   question  ───▶ │   FastAPI    │
                  │   /ask       │
                  └──────┬───────┘
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
      ┌───────────────┐     ┌──────────────┐
      │  PostgreSQL   │     │    Redis     │
      │  + pgvector   │     │   cache      │
      └───────────────┘     └──────────────┘
              │
              ▼
      ┌───────────────┐
      │  LLM (OpenAI- │
      │  compatible)  │
      └───────────────┘

   offline:  corpus ──▶ chunk ──▶ embed ──▶ pgvector
   CI:       golden set ──▶ evaluate ──▶ pass/fail gate
```

## Running it

```bash
cp .env.example .env     # add an LLM API key when retrieval lands
docker compose up --build
```

The API is then at `http://localhost:8000`, with interactive documentation at `http://localhost:8000/docs`.

### Local development

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows
pip install -e ".[dev]"

pytest
ruff check .
uvicorn app.main:app --reload
```

## Example request

```bash
curl http://localhost:8000/health
```

```json
{ "status": "ok", "version": "0.1.0", "environment": "local" }
```

## Roadmap

- [x] Service skeleton, containerisation, CI pipeline green from the first commit
- [x] Structure-aware chunking of regulatory text, with citations
- [x] Corpus fetch and parsing from the EU Publications Office
- [ ] Embeddings and pgvector storage
- [ ] Retrieval and cited answer generation
- [ ] Golden question set and evaluation harness
- [ ] Evaluation gate wired into CI
- [ ] Tracing, token and cost metrics
- [ ] Cloud deployment
- [ ] Retrieval strategy benchmark and written case study

## Architecture decisions

**PostgreSQL with pgvector rather than a dedicated vector database.** One datastore instead of two for a corpus of this size, which keeps operations and local setup simple. A dedicated vector store earns its place at a scale this project will not reach; the retrieval interface is kept narrow so it can be swapped if that assumption breaks.

**An OpenAI-compatible endpoint rather than a specific provider SDK.** Keeps the service portable across hosted and self-hosted models, and makes it possible to evaluate cheaper models against the same golden set.

**CI before features.** The evaluation gate is the point of this project, so the pipeline exists before there is anything to evaluate.

**The Publications Office Cellar service as the corpus source, not the EUR-Lex website.** The public EUR-Lex pages sit behind an AWS WAF that answers HTTP clients with `202` and `x-amzn-waf-action: challenge`, so no scripted fetch can retrieve them. Cellar serves the same documents through content negotiation at `publications.europa.eu/resource/celex/{celex}` and is the supported machine-readable interface. Downloads are cached on disk so repeat runs and tests never touch the network.

**Parsing the Official Journal's semantic classes rather than flattening to text.** Cellar XHTML marks articles as `p.oj-ti-art`, titles as `p.oj-sti-art` and divisions as `p.oj-ti-section-1`, which is far more reliable than pattern-matching prose. Two traps are worth knowing about: chapters and sections share the *same* CSS class and are distinguishable only by their heading text, and lettered points such as `(a)` are marked up as two-cell tables that must be rejoined to the paragraph they belong to.

**Structure-aware chunking rather than a fixed character window.** Regulations are already organised into chapters, articles and numbered paragraphs, so chunk boundaries follow that structure: short paragraphs merge, long ones split on sentence boundaries, and no chunk ever spans two articles. Two consequences follow. Every chunk maps to exactly one citation such as `Article 6(2)`, which is what makes grounded answers verifiable. And each chunk carries a breadcrumb header (`Article 9 > Risk management system > paragraph 1`) so that an embedded fragment retains the context a fixed-window splitter would have discarded.

**Character counts as the chunk budget, not tokens.** A tokeniser dependency buys precision this project does not yet need. If the budget starts mattering — for context-window packing or cost control — this is the first thing to revisit.

## Known limitations

- Not legal advice, and not a compliance tool. It answers questions about regulatory text; it does not interpret them for a specific situation.
- English-language source texts only.
- The golden question set is hand-written and therefore small; it catches regressions, it does not certify correctness.
- Recitals and annexes are not indexed. Recitals carry interpretive weight in EU law, so some questions are unanswerable by design until they are added.
- Source text is reproduced verbatim, including defects in the official publication. Article 1 of Regulation (EU) 2024/1689, for example, carries a stray backtick in its title upstream.

## License

MIT
