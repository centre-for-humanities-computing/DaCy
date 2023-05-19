import random
import warnings
from functools import partial
from typing import Dict, List, Optional

import altair as alt
import dacy
import numpy as np
import pandas as pd
import spacy
from datasets import load_dataset
from ner_bias_utils import (
    DACY_PIPES,
    SPACY_MODELS,
    SPACY_WRAP_MODELS,
    scandiner_loader,
    spacy_wrap_loader,
)
from spacy.scorer import Scorer
from spacy.tokens import Doc
from spacy.training import Example

DACY_MODELS = [
    "da_dacy_small_trf-0.2.0",
    "da_dacy_medium_trf-0.2.0",
    "da_dacy_large_trf-0.2.0",
]

ALL_MODELS_NAMES = SPACY_MODELS + DACY_MODELS + DACY_PIPES + SPACY_WRAP_MODELS

MDL_GETTER_DICT = {
    "da_core_news_sm-3.5.0": partial(spacy.load, "da_core_news_sm"),
    "da_core_news_md-3.5.0": partial(spacy.load, "da_core_news_md"),
    "da_core_news_lg-3.5.0": partial(spacy.load, "da_core_news_lg"),
    "da_core_news_trf-3.5.0": partial(spacy.load, "da_core_news_trf"),
    "da_dacy_small_trf-0.2.0": partial(dacy.load, "da_dacy_small_trf-0.2.0"),
    "da_dacy_medium_trf-0.2.0": partial(dacy.load, "da_dacy_medium_trf-0.2.0"),
    "da_dacy_large_trf-0.2.0": partial(dacy.load, "da_dacy_large_trf-0.2.0"),
    "da_dacy_small_ner_fine_grained-0.1.0": partial(
        dacy.load,
        "da_dacy_small_ner_fine_grained-0.1.0",
    ),
    "da_da_dacy_medium_ner_fine_grained-0.1.0": partial(
        dacy.load,
        "da_dacy_medium_ner_fine_grained-0.1.0",
    ),
    "da_dacy_large_ner_fine_grained-0.1.0": partial(
        dacy.load,
        "da_dacy_large_ner_fine_grained-0.1.0",
    ),
    "saattrupdan/nbailab-base-ner-scandi": scandiner_loader,
    "alexandrainst/da-ner-base": spacy_wrap_loader,
}


def dansk(splits: Optional[List[str]] = None, **kwargs):
    if splits is None:
        splits = ["train", "dev", "test"]

    if Doc.has_extension("meta"):
        warnings.warn("Overwriting existing meta extension")
    Doc.set_extension("meta", default={}, force=True)

    nlp = spacy.blank("da")

    def convert_to_doc(example):
        doc = Doc(nlp.vocab).from_json(example)
        # set metadata
        for k in ["dagw_source", "dagw_domain", "dagw_source_full"]:
            doc._.meta[k] = example[k]
        return doc

    return_ds = []
    for split in splits:
        ds = load_dataset("chcaa/DANSK", split=split, **kwargs)
        docs = [convert_to_doc(example) for example in ds]
        return_ds.append(docs)
    return return_ds


# convert to Conll-2003 format
def convert_to_conll_2003(
    docs,
    mapping={"PERSON": "PER", "GPE": "LOC", "LOCATION": "LOC", "ORGANIZATION": "ORG"},
) -> list:
    for doc in docs:
        ents = doc.ents
        ents = [e for e in ents if e.label_ in mapping]
        # convert GPE
        for ent in ents:
            ent.label_ = mapping[ent.label_]
        doc.ents = ents
    return docs


scorer = Scorer()


def no_misc_getter(doc, attr):
    for ent in doc.ents:
        if ent.label_ != "MISC":
            yield ent


def bootstrap(examples, n_rep=100):
    scores = []
    for _i in range(n_rep):
        sample = random.choices(examples, k=len(examples))
        score = scorer.score_spans(sample, getter=no_misc_getter, attr="ents")
        scores.append(score)
    return scores


def compute_mean_and_ci(scores):
    ent_f = [score["ents_f"] for score in scores]
    per_f = [score["ents_per_type"].get("PER", {"f": None})["f"] for score in scores]
    loc_f = [score["ents_per_type"].get("LOC", {"f": None})["f"] for score in scores]
    org_f = [score["ents_per_type"].get("ORG", {"f": None})["f"] for score in scores]

    nam = ["Average F1", "Person F1", "Location F1", "Organization F1"]

    d = {}
    for n, f in zip(nam, [ent_f, per_f, loc_f, org_f]):
        f = [x for x in f if x is not None]
        if len(f) == 0:
            d[n] = {"mean": None, "ci": None}
            continue
        d[n] = {"mean": np.mean(f), "ci": np.percentile(f, [2.5, 97.5])}
    return d


def evaluate_generalization(
    mdl_name,
    mdl,
    domains_dataset_dict=Dict[str, list],
) -> pd.DataFrame:
    rows = []
    all_examples = []
    for domain, docs in domains_dataset_dict.items():
        docs: list = domains_dataset_dict[domain]
        model_pred = mdl.pipe([doc.text for doc in docs])
        examples = [Example(predicted=x, reference=y) for x, y in zip(model_pred, docs)]
        all_examples.extend(examples)

        bs_score = bootstrap(examples)
        score = compute_mean_and_ci(bs_score)

        row = {
            "Model": mdl_name,
            "Domain": domain,
            "Average F1": score["Average F1"]["mean"],
            "Person F1": score["Person F1"]["mean"],
            "Location F1": score["Location F1"]["mean"],
            "Organization F1": score["Organization F1"]["mean"],
            "Average F1 CI": score["Average F1"]["ci"],
            "Number of docs": len(docs),
        }
        rows.append(row)

    # across domains
    examples = all_examples[mdl_name]
    bs_score = bootstrap(examples)
    score = compute_mean_and_ci(bs_score)

    row = {
        "Model": mdl_name,
        "Domain": "All",
        "Average F1": score["Average F1"]["mean"],
        "Person F1": score["Person F1"]["mean"],
        "Location F1": score["Location F1"]["mean"],
        "Organization F1": score["Organization F1"]["mean"],
        "Average F1 CI": score["Average F1"]["ci"],
        "Number of docs": len(examples),
    }
    rows.append(row)
    return pd.DataFrame(rows)


def create_generation_viz(df: pd.DataFrame):
    # filter out domains
    df = df[df["Domain"] != "danavis"]
    df = df[df["Domain"] != "dannet"]
    df = df[df["Domain"].notnull()]

    df["Average F1 CI Lower"] = df["Average F1 CI"].apply(lambda x: x[0])
    df["Average F1 CI Upper"] = df["Average F1 CI"].apply(lambda x: x[1])
    df["Average F1 CI Lower"] = pd.to_numeric(df["Average F1 CI Lower"])
    df["Average F1 CI Upper"] = pd.to_numeric(df["Average F1 CI Upper"])

    selection = alt.selection_point(
        fields=["Domain"],
        bind="legend",
        value=[{"Domain": "All"}],
    )  # does not work

    bind_checkbox = alt.binding_checkbox(
        name="Scale point size by number of documents: ",
    )
    param_checkbox = alt.param(bind=bind_checkbox)

    base = (
        alt.Chart(df)
        .mark_point(filled=True)
        .encode(
            # x='Average F1',
            x=alt.X("Average F1", title="F1"),
            y="Model",
            color="Domain",
            size=alt.condition(param_checkbox, "Number of docs", alt.value(100)),
            tooltip=[
                "Model",
                "Domain",
                "Average F1",
                "Person F1",
                "Location F1",
                "Organization F1",
            ],
            opacity=alt.condition(selection, alt.value(1), alt.value(0.0)),
        )
    )
    error_bars = (
        alt.Chart(df)
        .mark_errorbar(ticks=False)
        .encode(
            # x='Average F1 CI Lower',
            x=alt.X("Average F1 CI Lower", title="F1"),
            x2="Average F1 CI Upper",
            y="Model",
            color="Domain",
            opacity=alt.condition(selection, alt.value(1), alt.value(0.0)),
        )
    )

    chart = error_bars + base

    chart.add_params(selection, param_checkbox).properties(width=800, height=400)
    return chart
