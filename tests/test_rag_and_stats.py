from datetime import datetime, timedelta, timezone

from mlrca.rag import BM25Retriever, chunk_document, load_corpus, tokenize
from mlrca.telemetry import detect_mean_shift, psi

from .conftest import ROOT


def test_tokenize_splits_identifiers():
    toks = tokenize("amount_log_scaled in fraud-preproc")
    assert {"amount_log_scaled", "amount", "log", "scaled", "fraud-preproc", "preproc"} <= set(toks)


def test_chunking_by_heading_has_stable_ids():
    chunks = chunk_document("d1", {"title": "T", "type": "runbook"}, "intro\n\n# A\nalpha\n\n## B\nbeta\n")
    assert [c.chunk_id for c in chunks] == ["d1#overview", "d1#a", "d1#b"]
    assert all(c.doc_type == "runbook" for c in chunks)


def test_retrieval_finds_feature_contract_and_filters_by_type():
    r = BM25Retriever(load_corpus(ROOT / "data" / "corpus"))
    hits = r.search("amount_log_scaled training range transformation", top_k=3)
    assert "mc-fraud-detector-v12#feature-contract" in [h.chunk.chunk_id for h in hits]
    only_runbooks = r.search("rollback approval", doc_types=["runbook"])
    assert only_runbooks and all(h.chunk.doc_type == "runbook" for h in only_runbooks)


def test_psi_identical_is_zero_and_shift_is_large():
    assert psi([10, 20, 30], [10, 20, 30]) == 0
    assert psi([50, 40, 10], [5, 5, 90]) > 1


def test_mean_shift_detection():
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    pts = [(t0 + timedelta(hours=i), 0.9 if i < 30 else 0.7) for i in range(50)]
    cp = detect_mean_shift(pts)
    assert cp["index"] == 30 and cp["delta"] < -0.15
