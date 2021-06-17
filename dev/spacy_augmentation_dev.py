"""
This includes a series of SpaCy Augmenters
"""
from functools import partial
from math import inf
from typing import Callable, Iterator, List

import spacy
from spacy.training import Example
import random
from spacy.training.augment import create_lower_casing_augmenter
from spacy.language import Language


"""
Name sampling functions
"""


def sample_first_name(name: str, keep_name: bool, name_dict: dict) -> str:
    if keep_name:
        return name
    else:
        return random.choice(name_dict["first_name"])


def sample_abbreviation(name: str, keep_name: bool, name_dict: dict) -> str:
    if keep_name:
        return name[0]
    else:
        return random.choice(name_dict["first_name"])[0]


def sample_abbreviation_punct(name: str, keep_name: bool, name_dict: dict) -> str:
    if keep_name:
        return name[0] + "."
    else:
        return random.choice(name_dict["first_name"])[0] + "."


def sample_last_name(name: str, keep_name: bool, name_dict: dict) -> str:
    if keep_name:
        return name
    else:
        return random.choice(name_dict["last_name"])


"""
Slicers
"""


def get_ent_slices(entities: List[str], ent_type="PER") -> List[tuple]:
    slices = []

    start = None
    end = None
    for i, ent in enumerate(entities):
        if not ent.endswith(ent_type) and end == i - 1:
            slices.append(tuple([start, end + 1]))
        if ent.endswith(ent_type):
            if ent.startswith("U"):
                slices.append(tuple([i, i + 1]))
            if ent.startswith("B"):
                start = i
            if ent.startswith("L"):
                end = i
    return slices


def get_slice_spans(l: List[str], slices: List[tuple]):
    """Get the spans corresponding to some slices"""
    return [l[slice(s[0], s[1])] for s in slices]


"""
Augment
"""


def augment_entity(
    entities: List[List[str]],
    ent_dict: dict,
    pattern: str = "fn,ln",
    force_size: bool = False,
    keep_name: bool = False,
) -> List[List[str]]:
    """Augment entities

    Args:
        entities (List[List[str]]): A list of lists of spans of entities/names to replace
        pattern (str, optional): The pattern to replace with.
            Options: "fn", "ln", "abb", "abbpunct". Defaults to "fn,ln".
        keep_name (bool, optional): Whether to use the current name or sample from ent_dict

    Returns:
        List[List[str]]: [description]
    """
    pattern = pattern.split(",")

    new_entity_spans = []

    for entity_span in entities:
        if force_size:
            entity_span = resize_entity_list(entity_span, pattern)

        new_entity = []

        for i, ent in enumerate(entity_span):
            if i > len(pattern):
                continue
            if pattern[i] == "fn":
                new_entity.append(sample_first_name(ent, keep_name, ent_dict))
            if pattern[i] == "ln":
                new_entity.append(sample_last_name(ent, keep_name, ent_dict))
            if pattern[i] == "abb":
                new_entity.append(sample_abbreviation(ent, keep_name, ent_dict))
            if pattern[i] == "abbpunct":
                new_entity.append(sample_abbreviation_punct(ent, keep_name, ent_dict))
        new_entity_spans.append(new_entity)
    return new_entity_spans


def resize_entity_list(entity: List[str], pattern: List[str], ent_dict: dict):
    if len(entity) > len(pattern):
        return entity[: len(pattern)]
    else:
        return entity + [
            random.choice(ent_dict["last_name"])
            for _ in range(len(pattern) - len(entity))
        ]


def make_ent_dict():
    return {
        "first_name": ["Johnny", "Birte", "Tony"],
        "last_name": ["Johnnysen", "Birtesen", "Tonysen"],
    }


ent_dict = make_ent_dict()

nlp = spacy.load("da_core_news_sm")
doc = nlp("Mit navn er Kenneth Enevoldsen, og Lasse, Jakob og Kenneth.")
example = Example(doc, doc)
ex_dict = example.to_dict()

entity_slices = get_ent_slices(ex_dict["doc_annotation"]["entities"])
orth = get_slice_spans(ex_dict["token_annotation"]["ORTH"], entity_slices)

augment_entity(orth, ent_dict, force_size=True)
augment_entity(orth, ent_dict)
augment_entity(orth, ent_dict, pattern="abbpunct,ln,fn,ln,ln", force_size=False)
