import json
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import spacy
from spacy.language import Language
from spacy.scorer import Scorer
from spacy.tokens import Doc
from spacy.training import Example

from evaluation.datasets import datasets


def bootstrap(
    examples: List[Example],
    n_rep: int = 100,
    n_samples: Optional[int] = None,
    getter: Optional[Callable] = None,
) -> List[Dict[str, Any]]:
    random.seed(42)
    scorer = Scorer()
    scores = []
    if n_samples is None:
        n_samples = len(examples)
    for _i in range(n_rep):
        sample = random.choices(examples, k=n_samples)
        if getter is None:
            score = scorer.score_spans(sample, attr="ents")
        else:
            score = scorer.score_spans(sample, getter=getter, attr="ents")
        scores.append(score)
    return scores


def compute_mean_and_ci(scores: List[Dict[str, Any]]) -> Dict[str, Any]:
    ent_f = [score["ents_f"] for score in scores]
    # filter out None
    ent_f = [x for x in ent_f if x is not None]
    if ent_f:
        result_dict = {
            "Average": {
                "mean": np.mean(ent_f),
                "ci": np.percentile(ent_f, [2.5, 97.5]),
            },
        }
    else:
        result_dict = {"Average": {"mean": None, "ci": np.array([None, None])}}

    score_mapping = {
        "PER": "Person",
        "LOC": "Location",
        "LOCATION": "Location",
        "ORG": "Organization",
        "LANGUAGE": "Language",
        "PRODUCT": "Product",
        "LAW": "Law",
        "ORGANIZATION": "Organization",
        "WORK OF ART": "Work of Art",
        "PERSON": "Person",
        "FACILITY": "Facility",
        "GPE": "GPE",
        "EVENT": "Event",
        "CARDINAL": "Cardinal",
        "DATE": "Date",
        "MONEY": "Money",
        "NORP": "NORP",
        "ORDINAL": "Ordinal",
        "PERCENT": "Percent",
        "QUANTITY": "Quantity",
        "TIME": "Time",
        "MISC": "Misc.",
    }

    def get_ents_per_type(score):
        x = score["ents_per_type"]
        if x is None:
            return []
        return x

    labels = {label for score in scores for label in get_ents_per_type(score)}

    for label in labels:
        label_f = [
            score["ents_per_type"].get(label, {"f": None})["f"] for score in scores
        ]
        label_f = [x for x in label_f if x is not None]
        label = score_mapping.get(label, label)  # noqa
        if len(label_f) == 0:
            result_dict[label] = {"mean": None, "ci": None}
            continue
        result_dict[label] = {
            "mean": np.mean(label_f),
            "ci": np.percentile(label_f, [2.5, 97.5]),
        }
    return result_dict


def doc_to_json(doc: Doc) -> dict:
    json_obj = doc.to_json()
    if hasattr(doc._, "meta"):
        json_obj["meta"] = doc._.meta
    return json_obj


def doc_from_json(json_obj: dict, nlp: Language) -> Doc:
    doc = Doc(nlp.vocab).from_json(json_obj)
    if "meta" in json_obj:
        if not Doc.has_extension("meta"):
            Doc.set_extension("meta", default={}, force=True)
        doc._.meta = json_obj["meta"]
    return doc


def predictions_to_disk(
    save_path: Path,
    examples: List[Example],
    mdl_name: str,
    time_in_seconds: float,
) -> dict:
    save_path.parent.mkdir(exist_ok=True, parents=True)
    meta = {
        "mdl_name": mdl_name,
        "time_in_seconds": time_in_seconds,
        "Hardware": "Apple M1 Pro 16Gb running macOS 13.3.1",
    }

    # write to json
    meta["predicted"] = [doc_to_json(d.predicted) for d in examples]
    meta["reference"] = [doc_to_json(d.reference) for d in examples]

    with save_path.open("w") as f:
        json.dump(meta, f, indent=2)

    meta["examples"] = examples
    return meta


def predictions_from_disk(path: Path) -> dict:
    nlp = spacy.blank("da")
    with path.open() as f:
        meta = json.load(f)

    reference = [doc_from_json(d, nlp) for d in meta["reference"]]
    predicted = [doc_from_json(d, nlp) for d in meta["predicted"]]

    examples = []
    for ref, pred in zip(reference, predicted):
        example = Example(reference=ref, predicted=pred)
        examples.append(example)

    meta["examples"] = examples

    return meta


def apply_models(
    mdl_name: str,
    mdl_getter: Callable[[], Language],
    dataset: str,
    splits: list[str] = ["test"],  # noqa: B006
    cache: bool = True,
) -> dict:
    from time import time

    eval_path = Path(__file__).parent
    _mdl_name = mdl_name.replace("/", "_")
    save_folder = eval_path / "data" / f"{_mdl_name}"

    results = {}
    for split in splits:
        save_path = save_folder / f"{dataset}_{split}.json"
        if not save_path.exists() and cache:
            print(f"{dataset} ({split}): Running {mdl_name}")
            dataset_getter = datasets.get(dataset)
            examples = dataset_getter()[split]
            nlp = mdl_getter()

            start = time()
            docs = nlp.pipe(example.reference.text for example in examples)
            for doc, example in zip(docs, examples):
                example.predicted = doc
            end = time()
            time_in_seconds = end - start
            results = predictions_to_disk(
                save_path,
                examples,
                mdl_name,
                time_in_seconds,
            )
        else:
            print(f"{dataset} ({split}): Loading prediction for {mdl_name}")

        results[split] = predictions_from_disk(save_path)

    return results


def create_dataframe(
    examples: List[Example],
    mdl_name: str,
    decimals: int = 1,
    n_rep: int = 100,
    n_samples: Optional[int] = None,
) -> pd.DataFrame:
    score = bootstrap(examples, getter=None, n_rep=n_rep, n_samples=n_samples)
    score = compute_mean_and_ci(score)

    row = {
        "Models": mdl_name,
    }

    def score_to_string(score: Dict[str, Any], decimals: int = 1) -> str:
        if score["mean"] == 0:
            return " "
        return f"{100*score['mean']:.{decimals}f} ({100*score['ci'][0]:.{decimals}f}, {100*score['ci'][1]:.{decimals}f})"

    for key, value in score.items():
        row[key] = score_to_string(value, decimals=decimals)
    return pd.DataFrame([row])


def __string_repr(score: dict) -> str | None:
    if score["mean"] is None:
        return None
    return f"{score['mean']:.2f} ({score['ci'][0]:.2f}, {score['ci'][1]:.2f})"


def _create_row(
    mdl_name: str,
    domain: str,
    examples: list[Example],
    n_rep: int = 100,
    n_samples: Optional[int] = None,
):
    bs_score = bootstrap(examples, n_rep=n_rep, n_samples=n_samples)
    score = compute_mean_and_ci(bs_score)

    return {
        "Model": mdl_name,
        "Domain": domain,
        "Average": score["Average"]["mean"],
        "Average Lower CI": score["Average"]["ci"][0],
        "Average Upper CI": score["Average"]["ci"][1],
        "Average F1": __string_repr(score["Average"]),
        "Number of docs": len(examples),
    }


def evaluate_generalization(
    mdl_name: str,
    examples: list[Example],
    n_rep: int = 100,
    n_samples: Optional[int] = None,
    create_row_fn: Callable = _create_row,
) -> pd.DataFrame:
    domains = {}
    for example in examples:
        domain = example.y._.meta["dagw_domain"]
        if domain not in domains:
            domains[domain] = []
        domains[domain].append(example)

    rows = []
    for domain in domains:
        _examples = domains[domain]
        row = create_row_fn(
            mdl_name,
            domain,
            _examples,
            n_rep=n_rep,
            n_samples=n_samples,
        )
        rows.append(row)

    # all domains
    row = create_row_fn(mdl_name, "All", examples, n_rep=n_rep, n_samples=n_samples)
    rows.append(row)

    return pd.DataFrame(rows)


def convert_to_conll_2003(
    examples: list[Example],
    mapping: dict = {  # noqa: B006
        "PERSON": "PER",
        "GPE": "LOC",
        "LOCATION": "LOC",
        "ORGANIZATION": "ORG",
        "PER": "PER",
        "LOC": "LOC",
        "ORG": "ORG",
    },
) -> list:
    def doc_to_conll_2003(doc: Doc) -> Doc:
        ents = doc.ents
        ents = [e for e in ents if e.label_ in mapping]
        for ent in ents:
            ent.label_ = mapping[ent.label_]
        doc.ents = ents  # type: ignore
        return doc

    for example in examples:
        example.reference = doc_to_conll_2003(example.y)
        example.predicted = doc_to_conll_2003(example.x)
    return examples


def create_row_conll2003(
    mdl_name: str,
    domain: str,
    examples: list[Example],
    n_rep: int = 100,
    n_samples: Optional[int] = None,
) -> dict:
    def string_repr(score: dict) -> str | None:
        if score["mean"] is None:
            return None
        return f"{score['mean']:.2f} ({score['ci'][0]:.2f}, {score['ci'][1]:.2f})"

    bs_score = bootstrap(examples, n_rep=n_rep, n_samples=n_samples)
    score = compute_mean_and_ci(bs_score)

    orga = score.get("Organization", {"mean": None, "ci": (None, None)})
    person = score.get("Person", {"mean": None, "ci": (None, None)})
    location = score.get("Location", {"mean": None, "ci": (None, None)})

    return {
        "Model": mdl_name,
        "Domain": domain,
        "Average": score["Average"]["mean"],
        "Average Lower CI": score["Average"]["ci"][0],
        "Average Upper CI": score["Average"]["ci"][1],
        "Average F1": string_repr(score["Average"]),
        "Person F1": string_repr(person),
        "Organization F1": string_repr(orga),
        "Location F1": string_repr(location),
        "Number of docs": len(examples),
    }
