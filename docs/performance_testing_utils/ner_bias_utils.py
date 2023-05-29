## This create the table for bias scores

from functools import partial

import augmenty
import pandas as pd
import spacy
from spacy.language import Language
from spacy.training import Corpus, dont_augment

import dacy
from dacy.datasets import danish_names, female_names, male_names, muslim_names
from dacy.score import score

SPACY_MODELS = [
    "da_core_news_sm",
    "da_core_news_md",
    "da_core_news_lg",
    "da_core_news_trf",
]

DACY_MODELS = [
    "da_dacy_small_trf-0.2.0",
    "da_dacy_medium_trf-0.2.0",
    "da_dacy_large_trf-0.2.0",
]
DACY_MODELS_FINE = [
    "da_dacy_small_ner_fine_grained-0.1.0",
    "da_dacy_medium_ner_fine_grained-0.1.0",
    "da_dacy_large_ner_fine_grained-0.1.0",
]
DACY_PIPES = [
    "saattrupdan/nbailab-base-ner-scandi",
]

SPACY_WRAP_MODELS = [
    "alexandrainst/da-ner-base",
]

ALL_MODELS_NAMES = (
    SPACY_MODELS + DACY_MODELS + DACY_PIPES + SPACY_WRAP_MODELS + DACY_MODELS_FINE
)


def scandiner_loader():
    scandiner = spacy.blank("da")
    scandiner.add_pipe("dacy/ner")
    return scandiner


def spacy_wrap_loader(mdl):
    daner_base = spacy.blank("da")
    config = {"model": {"name": mdl}, "predictions_to": ["ents"]}
    daner_base.add_pipe("token_classification_transformer", config=config)
    return daner_base


@Language.component("conll2003_converter")
def conll2003_converter(doc):
    """
    converts ents to conllu format
    """
    mapping = {"PERSON": "PER", "ORGANIZATION": "ORG", "GPE": "LOC", "LOCATION": "LOC"}

    new_ents = []
    for ent in doc.ents:
        if ent.label_ in mapping:
            ent.label_ = mapping[ent.label_]
            new_ents.append(ent)
    doc.ents = new_ents
    return doc


def dacy_ner_mdl_fine_to_conll2003(mdl: str):
    nlp = dacy.load(mdl)
    # create component to convert annotations to conll2003
    nlp.add_pipe("conll2003_converter", last=True)
    return nlp


def no_misc_getter(doc, attr):
    for ent in doc.ents:
        if ent.label_ != "MISC":
            yield ent


MDL_GETTER_DICT = {
    "saattrupdan/nbailab-base-ner-scandi": scandiner_loader,
    "da_dacy_large_trf-0.2.0": partial(dacy.load, "da_dacy_large_trf-0.2.0"),
    "da_dacy_medium_trf-0.2.0": partial(dacy.load, "da_dacy_medium_trf-0.2.0"),
    "da_dacy_small_trf-0.2.0": partial(dacy.load, "da_dacy_small_trf-0.2.0"),
    "da_dacy_large_ner_fine_grained-0.1.0": partial(
        dacy_ner_mdl_fine_to_conll2003,
        "da_dacy_large_ner_fine_grained-0.1.0",
    ),
    "da_dacy_medium_ner_fine_grained-0.1.0": partial(
        dacy_ner_mdl_fine_to_conll2003,
        "da_dacy_medium_ner_fine_grained-0.1.0",
    ),
    "da_dacy_small_ner_fine_grained-0.1.0": partial(
        dacy_ner_mdl_fine_to_conll2003,
        "da_dacy_small_ner_fine_grained-0.1.0",
    ),
    "alexandrainst/da-ner-base": partial(
        spacy_wrap_loader,
        "alexandrainst/da-ner-base",
    ),
    "da_core_news_trf-3.5.0": partial(spacy.load, "da_core_news_trf"),
    "da_core_news_lg-3.5.0": partial(spacy.load, "da_core_news_lg"),
    "da_core_news_md-3.5.0": partial(spacy.load, "da_core_news_md"),
    "da_core_news_sm-3.5.0": partial(spacy.load, "da_core_news_sm"),
}





def apply_models(
    models: list,
    dataset: Corpus,
    augmenters: dict,
    n_rep: int = 20,
) -> pd.DataFrame:
    rows = []
    for mdl_name, nlp in models:
        # Evaluate

        out = score(
            dataset,
            apply_fn=nlp,
            augmenters=[dont_augment],
            k=1,
            score_fn=["ents"],
        )
        out["ents_f"] = out["ents_f"] * 100
        row = {
            "Model": f"{mdl_name}",
            "Augmenter": "Baseline",
            "F1": f"{out['ents_f'].mean():.2f}",
        }
        rows.append(row)

        for aug_name, aug in augmenters.items():
            out = score(
                dataset,
                apply_fn=nlp,
                augmenters=[aug],
                k=n_rep,
                score_fn=["ents"],
            )
            out["ents_f"] = out["ents_f"] * 100
            row = {
                "Model": f"{mdl_name}",
                "Augmenter": aug_name,
                "F1": f"{out['ents_f'].mean():.2f} ± {out['ents_f'].std():.2f}",
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    df2 = df.pivot(index="Model", columns="Augmenter", values="F1")

    df2 = df2.reset_index()
    return df2


def highlight_max(s: pd.Series) -> list:
    """Highlight the maximum in a Series with bold text."""
    # convert to str for comparison
    s = s.astype(str)
    is_max = s == s.max()
    return ["font-weight: bold" if v else "" for v in is_max]


def underline_second_max(s: pd.Series) -> list:
    """Underline the second maximum in a Series."""
    is_second_max = s == s.sort_values(ascending=False).iloc[1]
    return ["text-decoration: underline" if v else "" for v in is_second_max]


def create_table(  # noqa: ANN201
    result_df: pd.DataFrame,
    augmenters: dict,
):
    # replace index with range
    result_df.index = range(len(result_df))

    nam = [("", "Models"), ("", "Baseline")] + [
        ("Augmenter", aug_name) for aug_name, _ in augmenters.items()
    ]
    super_header = pd.MultiIndex.from_tuples(nam)
    result_df.columns = super_header

    s = result_df.style.apply(highlight_max, axis=0, subset=result_df.columns[1:])
    s = s.apply(underline_second_max, axis=0, subset=result_df.columns[1:])

    # round to 2 decimals the baseline
    s = s.format("{:.2f}", subset=result_df.columns[1:2])

    # Add a caption
    s = s.set_caption("F1 scores for the different models and augmenters")

    # Center the header and left align the model names
    s = s.set_properties(subset=result_df.columns[1:], **{"text-align": "right"})

    super_header_style = [
        {"selector": ".level0", "props": [("text-align", "center")]},
        {"selector": ".col_heading", "props": [("text-align", "center")]},
    ]
    # Apply the CSS style to the styler
    s = s.set_table_styles(super_header_style)
    s = s.set_properties(subset=[("", "Models")], **{"text-align": "left"})
    # remove the index
    s = s.hide(axis="index")
    return s
