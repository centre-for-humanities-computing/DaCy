from dacy.augmenters.keyboard import Keyboard
import random
from functools import partial
from typing import Dict, Iterable, Iterator, Callable, Union, List

import spacy
from spacy.language import Language
from spacy.training import Example

from .utils import make_text_from_orth

from .keyboard import KEYBOARDS, Keyboard


@spacy.registry.augmenters("synonym_augmenter.v1")
def create_synonym_augmenter(
    level: float, synonyms: dict
) -> Callable[[Language, Example], Iterator[Example]]:
    """Creates an augmenter swaps a token with its synonym based on a dictionary

    Args:
        level (float): Probability to replace token given that it is in synonym dictionary.
        synonyms (dict): a dictionary of words and a list of their synonyms

    Returns:
        Callable[[Language, Example], Iterator[Example]]: The augmenter function.
    """
    return partial(synonym_augmenter, level=level, synonyms=synonyms)

# TODO check if pos tags is available for synonyms repl

def synonym_augmenter(
    nlp: Language,
    example: Example,
    level: float,
    synonyms: dict,
) -> Iterator[Example]:
    def __replace(t):
        if t.text in synonyms and random.random() < level:
            return random.sample(synonyms[t.text], k=1)[0]
        return t

    example_dict = example.to_dict()
    example_dict["token_annotation"]["ORTH"] = [
        __replace(t) for t in example.reference
    ]
    text = make_text_from_orth(example_dict)
    doc = nlp.make_doc(text)
    yield example.from_dict(doc, example_dict)

@spacy.registry.augmenters("token_swap_augmenter.v1")
def create_token_swap_augmenter(
    level: float, respect_spans: bool = True
) -> Callable[[Language, Example], Iterator[Example]]:
    """Creates an augmenter that randomly swaps two neighbouring tokens.

    Args:
        level (float): The probability to swap two tokens.
        respect_spans (bool): Should the pipeline respect spans? Defaults to True. In which
        case it will not swap a token inside a span with a token outside the span.

    Returns:
        Callable[[Language, Example], Iterator[Example]]: The augmenter function.
    """
    return partial(token_swap_augmenter, level=level)


def token_swap_augmenter(
    nlp: Language,
    example: Example,
    level: float,
) -> Iterator[Example]:

    example_dict = example.to_dict()
    
    n_tok = len(example.y)
    for i in range(n_tok):
        if random.random() < level:
            # select which neighbour
            fb = random.sample([1, -1], k=1)[0]

            # TODO add check for spans

            n = i + fb if 0 < i + fb < n_tok else i - fb

    tok_anno = example_dict["token_annotation"]
    for k in tok_anno:
        tok_anno[k][i], tok_anno[k][n] = tok_anno[k][n], tok_anno[k][i] 
        # TODO fix broken spans

    text = make_text_from_orth(example_dict)
    doc = nlp.make_doc(text)
    yield example.from_dict(doc, example_dict)