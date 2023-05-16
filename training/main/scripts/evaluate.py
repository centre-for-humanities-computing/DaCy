import json
from pathlib import Path

import spacy
import typer
from spacy.scorer import Scorer
from spacy.training import Corpus, Example  # type: ignore
from spacy_experimental.coref.coref_scorer import (
    ClusterEvaluator,
    get_cluster_info,
    lea,
)
from wasabi import msg


def score_coref(examples):
    PREFIX = "coref_clusters"
    skipped_clusters = 0
    num_gold_clusters = 0
    num_pred_clusters = 0
    repeated_mentions = 0

    def example2clusters(example: Example):
        pred = []
        gold = []
        nonlocal skipped_clusters
        nonlocal num_gold_clusters
        nonlocal num_pred_clusters
        nonlocal repeated_mentions
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

    lea_evaluator = ClusterEvaluator(lea)
    for ex in examples:
        p_clusters, g_clusters = example2clusters(ex)
        cluster_info = get_cluster_info(p_clusters, g_clusters)
        lea_evaluator.update(cluster_info)

    return lea_evaluator


def apply_and_score(nlp, examples):
    docs = nlp.pipe(e.x.text for e in examples)
    for e, doc in zip(examples, docs):
        e.predicted = doc


def main(
    model_path: str,
    split: str = "test",
    gpu_id: int = -1,
    overwrite: bool = False,
):
    if gpu_id >= 0:
        spacy.require_gpu(gpu_id=gpu_id)

    model_name = Path(model_path).name
    project_path = Path(__file__).parent.parent
    corpus = project_path / Path("corpus")
    output_path = project_path / Path("metrics") / model_name
    if output_path.exists() and not overwrite:
        raise ValueError(
            "Output path already exists, set --overwrite to True to overwrite",
        )
    output_path.mkdir(parents=True, exist_ok=True)

    dane_path = corpus / "dane" / f"{split}.spacy"
    cdt_path = corpus / "cdt" / f"{split}.spacy"
    da_ddt_path = corpus / "da_ddt" / f"{split}.spacy"

    msg.info(f"Loading model from {model_path}")
    nlp = spacy.load(model_path)

    _dane = Corpus(dane_path)
    _cdt = Corpus(cdt_path)
    _da_ddt = Corpus(da_ddt_path)

    msg.info(f"Loading {split} split from datasets")
    dane = list(_dane(nlp))
    cdt = list(_cdt(nlp))
    da_ddt = list(_da_ddt(nlp))

    msg.info(f"Applying model to {split} split")
    apply_and_score(nlp, dane)
    apply_and_score(nlp, cdt)
    apply_and_score(nlp, da_ddt)

    lea = score_coref(cdt)
    scores_coref = {
        "coref_lea_f1": lea.get_f1(),
        "coref_lea_precision": lea.get_precision(),
        "coref_lea_recall": lea.get_recall(),
    }
    scorer = Scorer()
    scores = scorer.score(da_ddt)
    score_nel = scorer.score_links(cdt, negative_labels=["NIL", ""])
    score_ents = scorer.score_spans(dane, "ents")
    # score lemmatization
    # extract component
    if nlp.has_pipe("trainable_lemmatizer"):
        lemmatizer = nlp.get_pipe("trainable_lemmatizer")
        lemma_scores = lemmatizer.score(dane)
        scores.update(lemma_scores)

    scores.update(scores_coref)
    scores.update(score_nel)
    scores.update(score_ents)

    # remove cats_* keys
    scores = {k: v for k, v in scores.items() if not k.startswith("cats_")}

    with open(output_path / "scores.json", "w") as f:
        f.write(json.dumps(scores, indent=2))
    msg.good(f"Evaluation done. Scores written to {output_path}/scores.json")


if __name__ == "__main__":
    typer.run(main)
