"""
Danish specific SpaCy augmenters.
"""

from functools import partial
from typing import Callable, Iterator

import spacy
from spacy.language import Language
from spacy.training import Example

from .character import char_replace_augmenter


@spacy.registry.augmenters("æøå_augmenter.v1")
def create_æøå_augmenter(
    doc_level: float, char_level: float
) -> Callable[[Language, Example], Iterator[Example]]:
    """Augments æøå into their spelling variants ae, oe, aa.

    Args:
        doc_level (float): probability to augment document.
        char_level (float): probability to augment character, if document is augmented.

    Returns:
        Callable[[Language, Example], Iterator[Example]]: The desired augmenter.
    """
    replace_dict = {
        "æ": ["ae"],
        "ø": ["oe"],
        "å": ["aa"],
        "Æ": ["Ae"],
        "Ø": ["Oe"],
        "Å": ["Aa"],
    }
    return partial(
        char_replace_augmenter,
        replacement=replace_dict,
        doc_level=doc_level,
        char_level=char_level,
    )
