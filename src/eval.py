import argparse
import json
import zipfile
from pathlib import Path

from tqdm import tqdm

import bert_score

from src import config
from src.ingestion.embedder import get_device
from src.rag.engine import RagEngine


def load_dataset(zip_path: str | Path) -> dict[str, dict]:
    with zipfile.ZipFile(zip_path) as z:
        queries = json.loads(z.read("queries.json"))
        answers = json.loads(z.read("answers.json"))
        relevant_docs = json.loads(z.read("relevant_docs.json"))
        corpus = json.loads(z.read("corpus.json"))
        errors = json.loads(z.read("errors.json"))

    dataset = {}
    for uuid, question in queries.items():
        chunk_uuids = relevant_docs.get(uuid, [])
        relevant_texts = [corpus[c] for c in chunk_uuids if c in corpus]
        dataset[uuid] = {
            "question": question,
            "expected_answer": answers.get(uuid, ""),
            "relevant_chunk_texts": relevant_texts,
            "error": errors.get(uuid),
        }
    return dataset


def run_queries(dataset: dict, cache_path: str | Path, force: bool = False) -> dict:
    cache_path = Path(cache_path)
    cache: dict = {}
    if not force and cache_path.exists():
        cache = json.loads(cache_path.read_text())

    uncached = [uid for uid, entry in dataset.items() if uid not in cache and not entry["error"]]
    if not uncached:
        return cache

    engine = RagEngine()
    for uid in tqdm(uncached, desc="Querying RAG", unit="query"):
        try:
            result = engine.ask(dataset[uid]["question"], [])
            if result["answer"] == config.LLM_UNAVAILABLE_MSG:
                cache[uid] = {"answer": result["answer"], "chunks": [], "error": "ollama_unavailable"}
            else:
                cache[uid] = {"answer": result["answer"], "chunks": result["chunks"], "error": None}
        except Exception as e:
            cache[uid] = {"answer": "", "chunks": [], "error": f"query_failed: {e}"}
        cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2))

    return cache


def compute_metrics(dataset: dict, cache: dict) -> dict:
    valid_uuids = [
        uid for uid in dataset
        if not dataset[uid]["error"] and uid in cache and not cache[uid].get("error")
    ]

    per_query = []

    if valid_uuids:
        generated = [cache[uid]["answer"] for uid in valid_uuids]
        expected = [dataset[uid]["expected_answer"] for uid in valid_uuids]
        _, _, F1 = bert_score.score(
            generated, expected,
            model_type=config.EVAL_BERTSCORE_MODEL,
            verbose=False,
            device=get_device(),
        )

        def _token_overlap(a: str, b: str) -> float:
            wa, wb = set(a.split()), set(b.split())
            return len(wa & wb) / max(len(wa), 1)

        for i, uid in enumerate(valid_uuids):
            relevant_texts = dataset[uid]["relevant_chunk_texts"]
            retrieved_chunks = cache[uid]["chunks"]
            if relevant_texts:
                matched = sum(
                    1 for t in relevant_texts
                    if any(_token_overlap(t, c["text"]) > 0.5 for c in retrieved_chunks)
                )
                recall = matched / len(relevant_texts)
            else:
                recall = 0.0
            per_query.append({"uuid": uid, "f1": F1[i].item(), "recall": recall, "error": None})

    for uid in dataset:
        err = dataset[uid]["error"] or cache.get(uid, {}).get("error")
        if err:
            per_query.append({"uuid": uid, "f1": None, "recall": None, "error": err})

    n_valid = len(valid_uuids)
    n_errors = len(dataset) - n_valid
    mean_f1 = sum(e["f1"] for e in per_query if e["f1"] is not None) / n_valid if n_valid else 0.0
    mean_recall = sum(e["recall"] for e in per_query if e["recall"] is not None) / n_valid if n_valid else 0.0

    return {
        "summary": {
            "mean_f1": round(mean_f1, 4),
            "mean_recall": round(mean_recall, 4),
            "pass": mean_f1 >= config.EVAL_THRESHOLD,
            "threshold": config.EVAL_THRESHOLD,
            "n_queries": n_valid,
            "n_errors": n_errors,
        },
        "per_query": per_query,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Évalue le pipeline RAG avec BertScore F1")
    parser.add_argument("--no-cache", action="store_true", help="Force le recalcul de toutes les requêtes")
    args = parser.parse_args()

    Path("data").mkdir(exist_ok=True)

    dataset = load_dataset(config.EVAL_DATASET_ZIP)
    cache = run_queries(dataset, config.EVAL_CACHE_PATH, force=args.no_cache)
    metrics = compute_metrics(dataset, cache)

    Path(config.EVAL_RESULTS_PATH).write_text(json.dumps(metrics, ensure_ascii=False, indent=2))

    errors = {e["uuid"]: e["error"] for e in metrics["per_query"] if e["error"]}
    Path(config.EVAL_ERRORS_PATH).write_text(json.dumps(errors, ensure_ascii=False, indent=2))

    s = metrics["summary"]
    status = "✓" if s["pass"] else "✗"
    print(f"Evaluation complete — {s['n_queries']} queries ({s['n_errors']} errors excluded)")
    print(f"  Mean BertScore F1 : {s['mean_f1']:.3f}  {status} (threshold: {s['threshold']})")
    print(f"  Mean Recall@k     : {s['mean_recall']:.3f}")
    print(f"Results saved to {config.EVAL_RESULTS_PATH}")


if __name__ == "__main__":
    main()
