"""
List of models using for testing
"""

from functools import partial

import dacy
import spacy
from spacy.language import Language


def scandiner_loader() -> Language:
    scandiner = spacy.blank("da")
    scandiner.add_pipe("dacy/ner")
    return scandiner


def spacy_wrap_loader(mdl: str) -> Language:
    daner_base = spacy.blank("da")
    config = {"model": {"name": mdl}, "predictions_to": ["ents"]}
    daner_base.add_pipe("token_classification_transformer", config=config)
    return daner_base


def openai_model_loader_simple_ner(model: str) -> Language:
    nlp = spacy.blank("da")
    nlp.add_pipe(
        "llm",
        config={
            "task": {
                "@llm_tasks": "spacy.NER.v2",
                "labels": ["PERSON", "ORGANISATION", "LOCATION"],
                "label_definitions": {
                    "PERSON": "People, including fictional",
                    "ORGANISATION": "Companies, agencies, institutions, etc.",
                    "LOCATION": "Countries, cities, states, mountain ranges, bodies of water etc.",
                },
            },
            "backend": {
                "@llm_backends": "spacy.REST.v1",
                "api": "OpenAI",
                "config": {"model": model},
            },
        },
    )
    nlp.initialize()
    return nlp


def openai_model_loader_fine_ner(model: str) -> Language:
    nlp = spacy.blank("da")

    label_desc = {
        "PERSON": "People, including fictional",
        "NORP": "Nationalities or religious or political groups",
        "FACILITY": "Building, airports, highways, bridges, etc.",
        "ORGANIZATION": "Companies, agencies, institutions, etc.",
        "GPE": "Countries, cities, states.",
        "LOCATION": "Non-GPE locations, mountain ranges, bodies of water",
        "PRODUCT": "Vehicles, weapons, foods, etc. (not services)",
        "EVENT": "Named hurricanes, battles, wars, sports events, etc.",
        "WORK OF ART": "Titles of books, songs, etc.",
        "LAW": "Named documents made into laws",
        "LANGUAGE": "Any named language",
        "DATE": "Absolute or relative dates or periods",
        "TIME": "Times smaller than a day",
        "PERCENT": "Percentage",
        "MONEY": "Monetary values, including unit",
        "QUANTITY": "Measurements, as of weight or distance",
        "ORDINAL": '"first", "second"',
        "CARDINAL": "Numerals that do no fall under another type",
    }

    nlp.add_pipe(
        "llm",
        config={
            "task": {
                "@llm_tasks": "spacy.NER.v2",
                "labels": list(label_desc.keys()),
                "label_definitions": label_desc,
            },
            "backend": {
                "@llm_backends": "spacy.REST.v1",
                "api": "OpenAI",
                "config": {"model": model},
            },
        },
    )
    nlp.initialize()
    return nlp


MODELS = {
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
        "da_dacy_small_ner_fine_grained-0.1.0",
    ),
    "saattrupdan/nbailab-base-ner-scandi": scandiner_loader,
    "alexandrainst/da-ner-base": partial(
        spacy_wrap_loader,
        "alexandrainst/da-ner-base",
    ),
    "da_core_news_trf-3.5.0": partial(spacy.load, "da_core_news_trf"),
    "da_core_news_lg-3.5.0": partial(spacy.load, "da_core_news_lg"),
    "da_core_news_md-3.5.0": partial(spacy.load, "da_core_news_md"),
    "da_core_news_sm-3.5.0": partial(spacy.load, "da_core_news_sm"),
    "openai/gpt-3.5-turbo (02/05/23)": partial(
        openai_model_loader_simple_ner,
        model="gpt-3.5-turbo",
    ),
    "openai/gpt-4 (02/05/23)": partial(openai_model_loader_simple_ner, model="gpt-4"),
}
