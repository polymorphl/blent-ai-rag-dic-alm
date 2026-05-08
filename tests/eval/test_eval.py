import io, json, zipfile
import pytest
from src.eval import load_dataset
from src import config


def _make_zip(queries, answers, relevant_docs, corpus, errors):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("queries.json", json.dumps(queries))
        z.writestr("answers.json", json.dumps(answers))
        z.writestr("relevant_docs.json", json.dumps(relevant_docs))
        z.writestr("corpus.json", json.dumps(corpus))
        z.writestr("errors.json", json.dumps(errors))
    buf.seek(0)
    return buf


def test_load_dataset_joins_all_files():
    zf = _make_zip(
        queries={"uuid-1": "Quelle est la durée recommandée ?"},
        answers={"uuid-1": "5 ans"},
        relevant_docs={"uuid-1": ["chunk-1"]},
        corpus={"chunk-1": "Durée recommandée : 5 ans minimum."},
        errors={},
    )
    dataset = load_dataset(zf)

    assert "uuid-1" in dataset
    entry = dataset["uuid-1"]
    assert entry["question"] == "Quelle est la durée recommandée ?"
    assert entry["expected_answer"] == "5 ans"
    assert entry["relevant_chunk_texts"] == ["Durée recommandée : 5 ans minimum."]
    assert entry["error"] is None


def test_load_dataset_propagates_errors():
    zf = _make_zip(
        queries={"uuid-2": "question"},
        answers={"uuid-2": "réponse"},
        relevant_docs={"uuid-2": []},
        corpus={},
        errors={"uuid-2": "inference_error"},
    )
    dataset = load_dataset(zf)

    assert dataset["uuid-2"]["error"] == "inference_error"


def test_load_dataset_missing_corpus_chunk_is_skipped():
    zf = _make_zip(
        queries={"uuid-3": "question"},
        answers={"uuid-3": "réponse"},
        relevant_docs={"uuid-3": ["missing-chunk"]},
        corpus={},
        errors={},
    )
    dataset = load_dataset(zf)

    assert dataset["uuid-3"]["relevant_chunk_texts"] == []


import tempfile
from unittest.mock import MagicMock, patch


def test_run_queries_uses_cache(tmp_path):
    cache_path = tmp_path / "cache.json"
    existing = {"uuid-1": {"answer": "cached answer", "chunks": [], "error": None}}
    cache_path.write_text(json.dumps(existing))

    dataset = {
        "uuid-1": {"question": "q", "expected_answer": "a", "relevant_chunk_texts": [], "error": None}
    }

    with patch("src.eval.RagEngine") as MockEngine:
        from src.eval import run_queries
        result = run_queries(dataset, cache_path)
        MockEngine.assert_not_called()

    assert result["uuid-1"]["answer"] == "cached answer"


def test_run_queries_calls_engine_for_uncached(tmp_path):
    cache_path = tmp_path / "cache.json"
    dataset = {
        "uuid-2": {"question": "q2", "expected_answer": "a2", "relevant_chunk_texts": [], "error": None}
    }

    with patch("src.eval.RagEngine") as MockEngine:
        mock_instance = MockEngine.return_value
        mock_instance.ask.return_value = {
            "answer": "generated answer",
            "sources": [],
            "chunks": [{"text": "chunk text", "source": "doc.pdf", "score": 0.9, "metadata": {}}],
        }
        from src.eval import run_queries
        result = run_queries(dataset, cache_path)

    assert result["uuid-2"]["answer"] == "generated answer"
    assert result["uuid-2"]["chunks"] == [{"text": "chunk text", "source": "doc.pdf", "score": 0.9, "metadata": {}}]
    assert cache_path.exists()


def test_run_queries_marks_ollama_unavailable(tmp_path):
    cache_path = tmp_path / "cache.json"
    dataset = {
        "uuid-3": {"question": "q3", "expected_answer": "a3", "relevant_chunk_texts": [], "error": None}
    }

    with patch("src.eval.RagEngine") as MockEngine:
        mock_instance = MockEngine.return_value
        mock_instance.ask.return_value = {
            "answer": config.LLM_UNAVAILABLE_MSG,
            "sources": [],
            "chunks": [],
        }
        from src.eval import run_queries
        result = run_queries(dataset, cache_path)

    assert result["uuid-3"]["error"] == "ollama_unavailable"


def test_run_queries_skips_dataset_errors(tmp_path):
    cache_path = tmp_path / "cache.json"
    dataset = {
        "uuid-4": {"question": "q4", "expected_answer": "a4", "relevant_chunk_texts": [], "error": "inference_error"}
    }

    with patch("src.eval.RagEngine") as MockEngine:
        from src.eval import run_queries
        result = run_queries(dataset, cache_path)
        MockEngine.assert_not_called()

    assert "uuid-4" not in result


def test_recall_exact_substring_match():
    dataset = {"uuid-1": {"relevant_chunk_texts": ["texte pertinent"], "error": None, "expected_answer": "réponse"}}
    cache = {"uuid-1": {"answer": "réponse générée", "chunks": [{"text": "voici le texte pertinent du document"}], "error": None}}

    with patch("src.eval.bert_score.score") as mock_bs:
        import torch
        mock_bs.return_value = (None, None, torch.tensor([0.7]))
        from src.eval import compute_metrics
        metrics = compute_metrics(dataset, cache)

    assert metrics["per_query"][0]["recall"] == 1.0


def test_recall_token_overlap_match_without_substring():
    # Reference chunk is larger than the retrieved chunk (different chunking boundaries).
    # No substring match in either direction, but > 50% word overlap → should count as matched.
    reference = "La durée de placement recommandée est supérieure à cinq ans pour ce fonds"
    retrieved = "Durée recommandée : supérieure à cinq ans pour ce placement dans le fonds"
    dataset = {"uuid-1": {"relevant_chunk_texts": [reference], "error": None, "expected_answer": "réponse"}}
    cache = {"uuid-1": {"answer": "réponse", "chunks": [{"text": retrieved}], "error": None}}

    with patch("src.eval.bert_score.score") as mock_bs:
        import torch
        mock_bs.return_value = (None, None, torch.tensor([0.7]))
        from src.eval import compute_metrics
        metrics = compute_metrics(dataset, cache)

    assert metrics["per_query"][0]["recall"] == 1.0


def test_recall_no_match():
    dataset = {"uuid-1": {"relevant_chunk_texts": ["texte introuvable"], "error": None, "expected_answer": "réponse"}}
    cache = {"uuid-1": {"answer": "réponse", "chunks": [{"text": "contenu sans rapport"}], "error": None}}

    with patch("src.eval.bert_score.score") as mock_bs:
        import torch
        mock_bs.return_value = (None, None, torch.tensor([0.4]))
        from src.eval import compute_metrics
        metrics = compute_metrics(dataset, cache)

    assert metrics["per_query"][0]["recall"] == 0.0


def test_metrics_excludes_errors():
    dataset = {
        "uuid-ok": {"relevant_chunk_texts": [], "error": None, "expected_answer": "bonne réponse"},
        "uuid-err": {"relevant_chunk_texts": [], "error": "ollama_unavailable", "expected_answer": ""},
    }
    cache = {
        "uuid-ok": {"answer": "réponse", "chunks": [], "error": None},
        "uuid-err": {"answer": "", "chunks": [], "error": "ollama_unavailable"},
    }

    with patch("src.eval.bert_score.score") as mock_bs:
        import torch
        mock_bs.return_value = (None, None, torch.tensor([0.65]))
        from src.eval import compute_metrics
        metrics = compute_metrics(dataset, cache)

    assert metrics["summary"]["n_queries"] == 1
    assert metrics["summary"]["n_errors"] == 1
    err_entry = next(e for e in metrics["per_query"] if e["uuid"] == "uuid-err")
    assert err_entry["error"] == "ollama_unavailable"
    assert err_entry["f1"] is None


def test_metrics_pass_threshold():
    dataset = {"uuid-1": {"relevant_chunk_texts": [], "error": None, "expected_answer": "réponse"}}
    cache = {"uuid-1": {"answer": "réponse", "chunks": [], "error": None}}

    with patch("src.eval.bert_score.score") as mock_bs:
        import torch
        mock_bs.return_value = (None, None, torch.tensor([0.72]))
        from src.eval import compute_metrics
        metrics = compute_metrics(dataset, cache)

    assert metrics["summary"]["pass"] is True
    assert metrics["summary"]["mean_f1"] == pytest.approx(0.72, abs=1e-4)


def test_metrics_fail_threshold():
    dataset = {"uuid-1": {"relevant_chunk_texts": [], "error": None, "expected_answer": "réponse"}}
    cache = {"uuid-1": {"answer": "réponse", "chunks": [], "error": None}}

    with patch("src.eval.bert_score.score") as mock_bs:
        import torch
        mock_bs.return_value = (None, None, torch.tensor([0.45]))
        from src.eval import compute_metrics
        metrics = compute_metrics(dataset, cache)

    assert metrics["summary"]["pass"] is False
