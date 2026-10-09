import json
import pytest
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import context_precision, context_recall
from src.service.vectorstore import get_documents, get_documents_without_metadata


def run_retrieval(query: str) -> list[str]:   
    raw_results = get_documents_without_metadata(query)
    return [doc.page_content for doc, _score in raw_results]


def load_golden_dataset(json_file_path: str = "src/evaluation/golden_dataset.json") -> list[dict]: 
    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = []
    for doc_entry in data:
        doc_name = doc_entry.get("document", "Unknown")
        for item in doc_entry.get("golden_dataset", []):
            samples.append({
                "document": doc_name,
                "question": item["question"],
                "ground_truth": item["ground_truth"],
            })
    return samples


def test_retrieval_quality_batch():
    samples = load_golden_dataset()
    questions, contexts, ground_truths, doc_names = [], [], [], []

    for i, sample in enumerate(samples):
        q = sample["question"]
        retrieved = run_retrieval(q)
        questions.append(q)
        contexts.append(retrieved)
        ground_truths.append(sample["ground_truth"])
        doc_names.append(sample["document"])

    dataset = Dataset.from_dict({
        "question": questions,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    results = evaluate(
        dataset=dataset,
        metrics=[context_precision, context_recall],
    )

    precisions = results["context_precision"]
    recalls    = results["context_recall"]

    if not isinstance(precisions, list):
        precisions = [precisions]
    if not isinstance(recalls, list):
        recalls = [recalls]


    THRESHOLD = 0.7

    failures = []
    for i, (p, r) in enumerate(zip(precisions, recalls)):
        print(f"precisions : {p}, recalls : {r} ")
        passed = p >= THRESHOLD and r >= THRESHOLD
        if not passed:
            failures.append((i, questions[i], p, r))

    avg_precision = sum(precisions) / len(precisions)
    avg_recall    = sum(recalls)    / len(recalls)

    print(f"avg_precision : {avg_precision}")
    print(f"avg_recall : {avg_recall}")

    assert avg_precision > 0.8
    assert avg_recall > 0.8
    
    