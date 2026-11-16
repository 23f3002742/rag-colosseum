# RAG Colosseum — Vercel deployment

A self-contained FastAPI service for the 50 personalized technical passages. It exposes the required `POST /query` endpoint at the root of the deployed domain.

## Why this version fits Vercel

This project deliberately avoids PyTorch, FAISS native wheels, model downloads, remote embedding APIs, and API keys. At process initialization it builds a compact **256-dimensional corpus-fitted distributional embedding index** using deterministic random indexing over word co-occurrence. It combines dense cosine similarity with BM25-style lexical scoring and an extractive sentence reranker.

The retriever uses only the supplied 50 passages. Answers are extracted from passage sentences; retrieved text is never passed to a generative model or treated as executable instructions. This removes a common prompt-injection path. Requests are independent and read-only after initialization, so concurrent calls can share the in-memory index.

This is a compact corpus-fitted embedding model, not a pretrained transformer model. It is designed for this fixed 50-passage corpus and prioritizes reliable exact-fact retrieval, close-confounder discrimination, and abstention over open-domain answering.

## API contract

### `POST /query`

Request:

```json
{
  "question": "What quorum is required by Raft?",
  "top_k": 3
}
```

Response:

```json
{
  "answer": "Raft consensus requires a quorum of 4 nodes to commit a log entry.",
  "sources": [0],
  "scores": [0.88],
  "confidence": 0.92
}
```

The numerical score/confidence above is illustrative; actual values are computed by the retriever.

- `answer`: extractive answer string, or `null` when unsupported/off-topic.
- `sources`: zero-based passage indices (`0`–`49`) in descending relevance-score order.
- `scores`: same length/order as `sources`, clamped to `[0, 1]`.
- `confidence`: `[0, 1]`; abstentions are below `0.4`, supported answers above `0.6`.
- `top_k`: defaults to `3`, allowed range `1`–`10`.
- Wildcard CORS is returned, including preflight responses.

### `GET /health`

Returns readiness, passage count, and embedding dimensions. The index is constructed locally during Python module initialization; there is no remote model download.

### `GET /`

Returns a small service description.

## Deploy to Vercel

Prerequisites: a Vercel account and Node.js/npm installed locally.

1. Extract this ZIP file.
2. Open a terminal in the extracted `rag_colosseum_vercel` directory (the directory containing `app.py`, `passages.json`, `requirements.txt`, and `vercel.json`).
3. Install and sign in to the Vercel CLI:

   ```bash
   npm install -g vercel
   vercel login
   ```

4. Create a preview deployment:

   ```bash
   vercel
   ```

5. Deploy to production:

   ```bash
   vercel --prod
   ```

Accept the project setup prompts. Keep the project root set to this directory. Do not configure a custom build command or output directory; Vercel detects the root-level FastAPI `app` instance in `app.py` and installs `requirements.txt`.

The deployed routes are:

```text
https://YOUR-PROJECT.vercel.app/
https://YOUR-PROJECT.vercel.app/health
https://YOUR-PROJECT.vercel.app/query
```

Submit the base URL only, without `/query`.

## Verify the deployment

Health check:

```bash
curl https://YOUR-PROJECT.vercel.app/health
```

Query test:

```bash
curl -i -X POST https://YOUR-PROJECT.vercel.app/query \
  -H "Content-Type: application/json" \
  -d '{"question":"What quorum is required by Raft?","top_k":3}'
```

Run the included smoke-test suite against the live service:

```bash
python smoke_test.py https://YOUR-PROJECT.vercel.app
```

It checks representative factual retrieval, comparisons, abstention, source/score validity, CORS, and concurrent requests. Every check has an 8-second HTTP timeout.

## Local development

```bash
python -m venv .venv
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

In a second terminal:

```bash
python smoke_test.py http://127.0.0.1:8000
```

You can also use `vercel dev` after installing the Vercel CLI to exercise the project through Vercel's local development runtime.

## Deployment notes

- The project contains no secrets and requires no environment variables.
- The application is stateless after startup and can serve parallel requests from one immutable in-memory index.
- `vercel.json` allows up to 60 seconds at the platform level. This does **not** extend the exam grader's stricter 8-second request deadline.
- Verify the live service and run `smoke_test.py` immediately after deploying. This project cannot create a public URL in your Vercel account; Vercel assigns that URL when you deploy.
- The exact answerable/unanswerable boundary in a hidden evaluation cannot be guaranteed in advance. The included tests are a reproducible local/live smoke suite, not the examiner's private 20-probe result.

## Files

- `app.py`: FastAPI API, dense distributional indexing, BM25 hybrid retrieval, sentence reranking, extractive answers, abstention, and CORS.
- `passages.json`: all 50 supplied passages in their original order (index `0` is the first passage).
- `requirements.txt`: minimal Python dependencies.
- `vercel.json`: Python Function duration setting.
- `.python-version`: Python `3.12`.
- `smoke_test.py`: live/local API validation.
- `test_queries.json`: example evaluation queries.
