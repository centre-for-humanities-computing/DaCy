"""
List of models using for testing
"""

from functools import partial

import spacy

import dacy


def scandiner_loader():
    scandiner = spacy.blank("da")
    scandiner.add_pipe("dacy/ner")
    return scandiner


def spacy_wrap_loader(mdl):
    daner_base = spacy.blank("da")
    config = {"model": {"name": mdl}, "predictions_to": ["ents"]}
    daner_base.add_pipe("token_classification_transformer", config=config)
    return daner_base


MODELS = {
    "saattrupdan/nbailab-base-ner-scandi": scandiner_loader,
    "da_dacy_large_trf-0.2.0": partial(dacy.load, "da_dacy_large_trf-0.2.0"),
    "da_dacy_medium_trf-0.2.0": partial(dacy.load, "da_dacy_medium_trf-0.2.0"),
    "da_dacy_small_trf-0.2.0": partial(dacy.load, "da_dacy_small_trf-0.2.0"),
    "da_dacy_large_ner_fine_grained-0.1.0": partial(
        dacy.load,
        "da_dacy_large_ner_fine_grained-0.1.0",
    ),
    "da_dacy_medium_ner_fine_grained-0.1.0": partial(
        dacy.load,
        "da_dacy_medium_ner_fine_grained-0.1.0",
    ),
    "da_dacy_small_ner_fine_grained-0.1.0": partial(
        dacy.load,
        "da_dacy_small_ner_fine_grained",
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
