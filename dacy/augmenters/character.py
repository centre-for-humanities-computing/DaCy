import random
from functools import partial
from typing import Dict, Iterator, Set, Tuple

from spacy.language import Language
from spacy.training import Example


def make_char_swap_augmenter(doc_level: float, char_level: float):
    return partial(char_swap_augmenter, doc_level=doc_level, char_level=char_level)


def make_remove_spacing_augmenter(doc_level: float, spacing_level: float):
    return partial(
        remove_spacing_augmenter, doc_level=doc_level, spacing_level=spacing_level
    )


def make_random_replace_augmenter(
    char_level: float, doc_level: float, keyboard: str = "QWERTY_EN"
):
    from .keyboard import KEYBOARDS

    kb = KEYBOARDS[keyboard]
    replace_dict = {k: kb.all_keys() for k in kb.all_keys()}
    return partial(
        char_replace_augmenter,
        replacement=replace_dict,
        doc_level=doc_level,
        char_level=char_level,
    )


def char_replace_augmenter(
    nlp: Language,
    example: Example,
    replacement: dict,
    doc_level: float = 0.5,
    char_level: float = 0.1,
) -> Iterator[Example]:
    def __replace(c):
        if random.random() < char_level and c in replacement:
            return random.sample(replacement[c], k=1)
        return c

    if random.random() >= doc_level:
        yield example
    else:
        example_dict = example.to_dict()
        doc = nlp.make_doc(
            example.text
        )  # TODO ASK LASSE: should I regenerate this text?
        example_dict["token_annotation"]["ORTH"] = [
            __replace(c) for t in example.reference for c in t
        ]
        yield example.from_dict(doc, example_dict)


def char_swap_augmenter(
    nlp: Language,
    example: Example,
    doc_level: float = 0.5,
    char_level: float = 0.1,
) -> Iterator[Example]:
    def __replace(t):
        for i, c in enumerate(t[:-1]):
            if random.random() < char_level:
                return t[:i] + t[i + 1] + c + t[i + 2 :]
        return t

    if random.random() >= doc_level:
        yield example
    else:
        example_dict = example.to_dict()
        doc = nlp.make_doc(
            example.text
        )  # TODO ASK LASSE: should I regenerate this text?
        example_dict["token_annotation"]["ORTH"] = [
            __replace(c) for t in example.reference for c in t
        ]
        yield example.from_dict(doc, example_dict)


def remove_spacing_augmenter(
    nlp: Language,
    example: Example,
    doc_level: float = 0.5,
    spacing_level: float = 0.1,
) -> Iterator[Example]:
    def __replace(s):
        if random.random() < spacing_level and (s is True):
            return False
        return s

    if random.random() >= doc_level:
        yield example
    else:
        example_dict = example.to_dict()
        doc = nlp.make_doc(
            example.text
        )  # TODO ASK LASSE: should I regenerate this text?
        example_dict["token_annotation"]["SPACY"] = [
            __replace(s) for s in example_dict["token_annotation"]["SPACY"]
        ]
        yield example.from_dict(doc, example_dict)
