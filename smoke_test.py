"""Smoke-test the deployed/local RAG API.

Usage:
    python smoke_test.py http://127.0.0.1:8000
    python smoke_test.py https://your-project.vercel.app
"""
from __future__ import annotations

import concurrent.futures
import json
import sys
import urllib.error
import urllib.request

BASE_URL = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000").rstrip("/")

CASES = [
    ("How many nodes must acknowledge a Raft log entry?", "4 nodes", 0),
    ("How long is the Multi-Paxos leader lease?", "500ms", 1),
    ("What is the initial learning rate for AdamW?", "0.01", 11),
    ("How many samples are used in the AdamW mini-batches?", "64 samples", 11),
    ("How does PostgreSQL clean up obsolete row versions?", "Vacuum", 20),
    ("What default port does MySQL use?", "6379", 21),
    ("What UDP port does QUIC use?", "5432", 31),
    ("What timeout applies to recursive DNS lookups?", "250ms", 32),
    ("What key derivation algorithm is memory-hard?", "Argon2", 42),
    ("What are the components of a JSON Web Token?", "header, payload", 49),
]
COMPARISON_CASES = [
    ("Compare consistent hashing and rendezvous hashing.", ["virtual ring", "highest-ranking server"], {6, 7}),
    ("Compare batch normalization and layer normalization.", ["mini-batch dimensions", "feature channel dimensions"], {18, 19}),
    ("Compare SGD and AdamW optimizers.", ["stochastic gradient descent", "decoupling weight decay"], {10, 11}),
    ("Compare PostgreSQL MVCC with MySQL InnoDB undo logs.", ["page-level multi-version", "undo log segments"], {20, 21}),
    ("Compare TLS 1.2 and TLS 1.3 handshake round trips.", ["two-round-trip", "single round trip"], {36, 37}),
    ("Compare HTTP/2 and HTTP/3 transport.", ["single TCP connection", "QUIC over UDP"], {34, 35}),
]
ABSTAIN_CASES = [
    "Who invented the Raft consensus algorithm?",
    "What year was PostgreSQL first released?",
    "What is the capital of France?",
    "What is the time complexity of Raft elections?",
]


def request(path: str, payload: dict | None = None, method: str | None = None, headers: dict | None = None):
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        BASE_URL + path,
        data=body,
        method=method or ("POST" if body is not None else "GET"),
        headers={**({"Content-Type": "application/json"} if body is not None else {}), **(headers or {})},
    )
    with urllib.request.urlopen(req, timeout=8) as response:
        raw = response.read().decode("utf-8")
        content_type = response.headers.get("Content-Type", "")
        data = json.loads(raw) if raw and "json" in content_type.lower() else None
        return response.status, {key.lower(): value for key, value in response.headers.items()}, data


def main() -> int:
    try:
        status, _headers, health = request("/health")
    except Exception as exc:
        print(f"FAIL /health: {exc}")
        return 1
    assert status == 200 and health.get("ready") is True and health.get("passages") == 50, health
    print("PASS health: 50 passages ready")

    for question, expected, source_id in CASES:
        status, headers, data = request("/query", {"question": question, "top_k": 3})
        assert status == 200, (question, status)
        assert headers.get("access-control-allow-origin") == "*", headers
        assert isinstance(data.get("answer"), str) and expected.lower() in data["answer"].lower(), (question, data)
        assert source_id in data["sources"], (question, data)
        assert len(data["sources"]) == len(data["scores"]), data
        assert data["scores"] == sorted(data["scores"], reverse=True), data
        assert all(0 <= i < 50 for i in data["sources"]), data
        assert all(0.0 <= s <= 1.0 for s in data["scores"]), data
        assert 0.6 < data["confidence"] <= 1.0, data
        print(f"PASS query: {question}")

    for question, expected_phrases, expected_sources in COMPARISON_CASES:
        status, headers, data = request("/query", {"question": question, "top_k": 3})
        assert status == 200, (question, status)
        answer = (data.get("answer") or "").lower()
        assert all(phrase.lower() in answer for phrase in expected_phrases), (question, data)
        assert expected_sources.issubset(set(data["sources"])), (question, data)
        assert data["scores"] == sorted(data["scores"], reverse=True), data
        print(f"PASS comparison: {question}")

    for question in ABSTAIN_CASES:
        _status, _headers, data = request("/query", {"question": question, "top_k": 3})
        assert data["answer"] in (None, "") and data["sources"] == [] and data["scores"] == [], (question, data)
        assert data["confidence"] < 0.4, (question, data)
        print(f"PASS abstention: {question}")

    _status, cors_headers, _data = request(
        "/query", method="OPTIONS", headers={
            "Origin": "https://example.com",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert cors_headers.get("access-control-allow-origin") == "*", cors_headers
    print("PASS CORS preflight")

    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        futures = [pool.submit(request, "/query", {"question": "What quorum is required by Raft?", "top_k": 3}) for _ in range(24)]
        results = [future.result() for future in futures]
    assert all(status == 200 and body["answer"] for status, _headers, body in results)
    print("PASS concurrent requests: 24 requests / 12 workers")
    print("ALL SMOKE TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
