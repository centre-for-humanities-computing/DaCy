import json

import spacy
import tqdm
import typer
from spacy.tokens import DocBin
from spacy.training import Example
from spacy_experimental.coref.coref_scorer import (
    ClusterEvaluator,
    get_cluster_info,
    lea,
)

PREFIX = "coref_clusters"
skipped_clusters = 0
num_gold_clusters = 0
num_pred_clusters = 0
repeated_mentions = 0


def example2clusters(example: Example):
    pred = []
    gold = []
    global skipped_clusters
    global num_gold_clusters
    global num_pred_clusters
    global repeated_mentions
    all_mentions = set()

    for name, span_group in example.predicted.spans.items():
        if not name.startswith(PREFIX):
            continue
        num_pred_clusters += 1
        aligned = example.get_aligned_spans_x2y(span_group)
        if not aligned:
            skipped_clusters += 1
            continue
        cluster = []
        for mention in aligned:
            cluster.append((mention.start, mention.end))
            if (mention.start, mention.end) in all_mentions:
                repeated_mentions += 1
            all_mentions.add((mention.start, mention.end))
        pred.append(cluster)

    for name, span_group in example.reference.spans.items():
        if not name.startswith(PREFIX):
            continue

        cluster = []
        num_gold_clusters += 1
        for mention in span_group:
            cluster.append((mention.start, mention.end))
        gold.append(cluster)
    return pred, gold


def main(model: str, dataset: str, gpu_id: int = -1):
    """
    Evaluate a coreference model on a dataset.

    Args:
        model: Path to the trained pipeline.
        dataset: Path to the test data.
        gpu_id: ID of GPU to run coreference pipeline on. Defaults to -1 (CPU only).
    """
    if gpu_id > -1:
        spacy.require_gpu(gpu_id)  # type: ignore

    # output
    nlp = spacy.load(model)
    gold_db = DocBin().from_disk(dataset)
    gold_docs = gold_db.get_docs(nlp.vocab)
    lea_evaluator = ClusterEvaluator(lea)
    for gold_doc in tqdm.tqdm(gold_docs):
        if len(gold_doc) == 0:
            print("WARNING: empty doc")
            continue
        pred_doc = nlp(gold_doc.text)
        ex = Example(predicted=pred_doc, reference=gold_doc)
        p_clusters, g_clusters = example2clusters(ex)
        cluster_info = get_cluster_info(p_clusters, g_clusters)
        lea_evaluator.update(cluster_info)

    print("LEA", lea_evaluator.get_f1())
    print("Gold clusters: ", num_gold_clusters)
    print("Predicted clusters: ", num_pred_clusters)
    print("Skipped predicted clusters: ", skipped_clusters)
    print("Repeated mentions: ", repeated_mentions)

    metrics = {
        "LEA_f1": lea_evaluator.get_f1(),
        "LEA_precision": lea_evaluator.get_precision(),
        "LEA_recall": lea_evaluator.get_recall(),
    }
    # write to metrics/evaluate_coref.json
    with open("evaluate_coref.json", "w") as f:
        json.dump(metrics, f)


if __name__ == "__main__":
    typer.run(main)
