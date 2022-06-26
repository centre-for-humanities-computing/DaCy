"""python -m spacy train config.cfg --code functions.py."""

import sys
from typing import Callable, Iterable, Iterator

import spacy

sys.path.append("../..")

from spacy.language import Language
from spacy.training import Example
from spacy.training.augment import create_lower_casing_augmenter

from dacy.augmenters import (
    create_char_swap_augmenter,
    create_keyboard_augmenter,
    create_pers_augmenter,
    create_æøå_augmenter,
)
from dacy.datasets.names import load_names


def combine_augmenters(
    augmenters: Iterable[Callable[[Language, Example], Iterator[Example]]],
) -> Callable[[Language, Example], Iterator[Example]]:
    """Combines a series og spaCy style augmenters.

    Args:
        augmenters (Iterable[Callable[[Language, Example], Iterator[Example]]]): An list of spaCy augmenters.

    Returns:
        Callable[[Language, Example], Iterator[Example]]: The combined augmenter
    """

    def apply_multiple_augmenters(nlp: Language, example: Example):
        examples = [example]
        for aug in augmenters:
            examples = (e for example in examples for e in aug(nlp, example))
        for e in examples:
            yield example

    return apply_multiple_augmenters


@spacy.registry.augmenters("dacy_augmenter.v1")
def dacy_augmenters():
    augmenters = [
        create_keyboard_augmenter(doc_level=1, char_level=0.02, distance=1.5),
        create_pers_augmenter(
            ent_dict=load_names(),
            patterns=[
                "fn,ln",
                "abbpunct,ln",
                "fn,ln,ln",
                "fn,ln,ln",
                "abb,ln",
                "ln,abbpunct",
            ],
            force_pattern_size=True,
            keep_name=False,
            prob=0.1,
        ),
        create_char_swap_augmenter(doc_level=1, char_level=0.02),
        create_æøå_augmenter(doc_level=1, char_level=0.1),
        create_lower_casing_augmenter(level=0.1),
    ]
    return combine_augmenters(augmenters)
