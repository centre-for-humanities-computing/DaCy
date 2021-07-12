"""
SpaCy augmenters for character level augmentation.
"""


from dacy.augmenters.keyboard import Keyboard
import random
from functools import partial
from typing import Dict, Iterable, Iterator, Callable, Union

import spacy
from spacy.language import Language
from spacy.training import Example

from .utils import make_text_from_orth

from .keyboard import KEYBOARDS, Keyboard


@spacy.registry.augmenters("char_swap_augmenter.v1")
def create_char_swap_augmenter(
    doc_level: float, char_level: float
) -> Callable[[Language, Example], Iterator[Example]]:
    """Creates an augmenter that swaps two characters in a token.

    Args:
        doc_level (float): probability to augment document.
        char_level (float): probability to augment character, if document is augmented.

    Returns:
        Callable[[Language, Example], Iterator[Example]]: The augmenter function.
    """
    return partial(char_swap_augmenter, doc_level=doc_level, char_level=char_level)


@spacy.registry.augmenters("spacing_augmenter.v1")
def create_spacing_augmenter(
    doc_level: float, spacing_level: float
) -> Callable[[Language, Example], Iterator[Example]]:
    """Creates an augmenter that removes spacing.

    Args:
        doc_level (float): probability to augment document.
        spacing_level (float): probability to remove spacing, if document is augmented.

    Returns:
        Callable[[Language, Example], Iterator[Example]]: The augmenter function.
    """
    return partial(spacing_augmenter, doc_level=doc_level, spacing_level=spacing_level)


@spacy.registry.augmenters("char_random_augmenter.v1")
def create_char_random_augmenter(
    doc_level: float, char_level: float, keyboard: Union[str, Keyboard] = "QWERTY_EN"
) -> Callable[[Language, Example], Iterator[Example]]:
    """Creates an augmenter that replacies a character with a random character from the
    keyboard.

    Args:
        doc_level (float): probability to augment document.
        char_level (float): probability to augment character, if document is augmented.
        keyboard (str, Keyboard, optional): A Keyboard class or a string denoting a default keyboard from
            which replace characters are sampled from. Possible options for string include:
            "QWERTY_EN": English QWERTY keyboard
            "QWERTY_DA": Danish QWERTY keyboard
            Defaults to "QWERTY_EN".

    Returns:
        Callable[[Language, Example], Iterator[Example]]: The augmenter function.
    """

    kb = Keyboard(keyboard_array=KEYBOARDS[keyboard])
    replace_dict = {k: list(kb.all_keys()) for k in kb.all_keys()}
    return partial(
        char_replace_augmenter,
        replacement=replace_dict,
        doc_level=doc_level,
        char_level=char_level,
    )


@spacy.registry.augmenters("keyboard_augmenter.v1")
def create_keyboard_augmenter(
    doc_level: float,
    char_level: float,
    distance=1,
    keyboard: Union[str, Keyboard] = "QWERTY_EN",
) -> Callable[[Language, Example], Iterator[Example]]:
    """Creates a document level augmenter using plausible typos based on keyboard distance.

    Args:
        doc_level (float): probability to augment document.
        char_level (float): probability to augment character, if document is augmented.
        distance (int, optional): keyboard distance. Defaults to 1.
        keyboard (str, Keyboard, optional): A Keyboard class or a string denoting a default keyboard.
            Possible options for string include:
            "QWERTY_EN": English QWERTY keyboard
            "QWERTY_DA": Danish QWERTY keyboard
            Defaults to "QWERTY_EN".

    Returns:
        Callable[[Language, Example], Iterator[Example]]: The augmentation function
    """
    kb = Keyboard(keyboard_array=KEYBOARDS[keyboard])
    replace_dict = kb.create_distance_dict(distance=distance)
    return partial(
        char_replace_augmenter,
        replacement=replace_dict,
        doc_level=doc_level,
        char_level=char_level,
    )


@spacy.registry.augmenters("char_replace_augmenter.v1")
def create_char_replace_augmenter(
    doc_level: float, char_level: float, replacement: dict
) -> Callable[[Language, Example], Iterator[Example]]:
    """Creates an augmenter that replaces a character with a random character from the
    keyboard.

    Args:
        doc_level (float): probability to augment document.
        char_level (float): probability to augment character, if document is augmented.
        replace (dict): A dictionary denoting which characters denote potentials replacement for each character.
            E.g. {"æ": "ae"}

    Returns:
        Callable[[Language, Example], Iterator[Example]]: The augmenter function.
    """
    return partial(
        char_replace_augmenter,
        replacement=replacement,
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
    def __replace(t):
        t_ = []
        for i, c in enumerate(t.text):
            if random.random() < char_level and c in replacement:
                c = random.sample(replacement[c], k=1)[0]
            t_.append(c)
        return "".join(t_)

    if random.random() >= doc_level:
        yield example
    else:
        example_dict = example.to_dict()
        example_dict["token_annotation"]["ORTH"] = [
            __replace(t) for t in example.reference
        ]
        text = make_text_from_orth(example_dict)
        doc = nlp.make_doc(text)
        yield example.from_dict(doc, example_dict)


def char_swap_augmenter(
    nlp: Language,
    example: Example,
    doc_level: float = 0.5,
    char_level: float = 0.1,
) -> Iterator[Example]:
    def __replace(t):
        for i, c in enumerate(t.text[:-1]):
            if random.random() < char_level:
                return t.text[:i] + t.text[i + 1] + c + t.text[i + 2 :]
        return t

    if random.random() >= doc_level:
        yield example
    else:
        example_dict = example.to_dict()
        example_dict["token_annotation"]["ORTH"] = [
            __replace(t) for t in example.reference
        ]
        text = make_text_from_orth(example_dict)
        doc = nlp.make_doc(text)
        yield example.from_dict(doc, example_dict)


def spacing_augmenter(
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
        example_dict["token_annotation"]["SPACY"] = [
            __replace(s) for s in example_dict["token_annotation"]["SPACY"]
        ]
        text = make_text_from_orth(example_dict)
        doc = nlp.make_doc(text)
        yield example.from_dict(doc, example_dict)
