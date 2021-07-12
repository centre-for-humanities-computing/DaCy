"""
This includes function for scoring models applied to a SpaCy corpus.
"""
from __future__ import annotations

from time import time
from copy import copy
from functools import partial
from typing import Callable, Dict, Iterable, List, Optional, Union

import pandas as pd
from spacy.language import Language
from spacy.scorer import Scorer
from spacy.training import Corpus, Example, dont_augment
from spacy.tokens import Doc, Span

from ..utils import flatten_dict


def no_misc_getter(doc: Doc, attr: str) -> Iterable[Span]:
    """A utility getter for scoring entities without including MISC

    Args:
        doc (Doc): a SpaCy Doc
        attr (str): attribute to be extracted

    Returns:
        Iterable[Span]
    """
    spans = getattr(doc, attr)
    for span in spans:
        if span.label_ == "MISC":
            continue
        yield span


def dep_getter(token, attr):
    dep = getattr(token, attr)
    dep = token.vocab.strings.as_string(dep).lower()
    return dep


def score(
    corpus: Corpus,
    apply_fn: Union[Callable[[Iterable[Example], List[Example]]], Language],
    score_fn: List[Union[Callable[[Iterable[Example]], dict], str]] = [
        "token",
        "pos",
        "ents",
        "dep",
    ],
    augmenters: List[Callable[[Language, Example], Iterable[Example]]] = [],
    k: int = 1,
    nlp: Optional[Language] = None,
    **kwargs,
) -> pd.DataFrame:
    """scores a models performance on a given corpus with potentially augmentations applied to it.

    Args:
        corpus (Corpus): A spacy Corpus
        apply_fn (Union[Callable, Language]): A wrapper function for the model you wish to score. The model should
            take in a list of spacy Examples (Iterable[Example]) and output a tagged version of it (Iterable[Example]).
            A SpaCy pipeline (Language) can be provided as is.
        score_fn (List[Union[Callable[[Iterable[Example]], dict], str]], optional): A scoring function which takes in a list of
            examples (Iterable[Example]) and return a dictionary of performance scores. Four potiential
            strings are valid. "ents" for measuring the performance of entity spans. "pos" for measuring
            the performance of fine-grained (tag_acc), and coarse-grained (pos_acc) pos-tags. "token" for measuring
            the performance of tokenization. "dep" for measuring the performance of dependency parsing. "nlp" for measuring
            the performance of all components in the specified nlp pipeline. Defaults to ["token", "pos", "ents", "dep"].
        augmenters (List[Callable[[Language, Example], Iterable[Example]]], optional): A spaCy style augmenters
            which should be applied to the corpus or a list thereof. defaults to [], indicating no augmenters.
        k (int, optional): Number of times it should run the augmentation and test the performance on
            the corpus. Defaults to 1.
        nlp (Optional[Language], optional): A spacy processing pipeline. If None it will use an empty
            Danish pipeline. Defaults to None. Used for loading the calling the corpus.

    Returns:
        pandas.DataFrame: returns a pandas dataframe containing the performance metrics.

    Example:
        >>> from spacy.training.augment import create_lower_casing_augmenter
        >>> from dacy.datasets import dane
        >>> test = dane(splits=["test")
        >>> nlp = dacy.load("da_dacy_small_tft-0.0.0")
        >>> scores = score(test, augmenter=[create_lower_casing_augmenter(0.5)], apply_fn = nlp)
    """
    if callable(augmenters):
        augmenters = [augmenters]
    if len(augmenters) == 0:
        augmenters = [dont_augment]

    def __apply_nlp(examples):
        examples = ((e.x.text, e.y) for e in examples)
        doc_tuples = nlp_.pipe(examples, as_tuples=True)
        return [Example(x, y) for x, y in doc_tuples]

    if isinstance(apply_fn, Language):
        nlp_ = apply_fn
        apply_fn = __apply_nlp

    if nlp is None:
        from spacy.lang.da import Danish

        nlp = Danish()

    scorer = Scorer(nlp)

    def ents_scorer(examples):
        scores = Scorer.score_spans(examples, attr="ents")
        scores_no_misc = Scorer.score_spans(
            examples, attr="ents", getter=no_misc_getter
        )
        scores["ents_excl_MISC"] = {
            k: scores_no_misc[k] for k in ["ents_p", "ents_r", "ents_f"]
        }
        return scores

    def pos_scorer(examples):
        scores = Scorer.score_token_attr(examples, attr="pos")
        scores_ = Scorer.score_token_attr(examples, attr="tag")
        for k in scores_:
            scores[k] = scores_[k]
        return scores

    def_scorers = {
        "ents": ents_scorer,
        "pos": pos_scorer,
        "token": Scorer.score_tokenization,
        "nlp": scorer.score,
        "dep": partial(
            Scorer.score_deps,
            attr="dep",
            getter=dep_getter,
            ignore_labels=("p", "punct"),
        ),
    }

    def __score(augmenter):

        corpus_ = copy(corpus)
        corpus_.augmenter = augmenter
        scores_ls = []
        for i in range(k):
            s = time()
            examples = apply_fn(corpus_(nlp))
            speed = time() - s
            scores = {"wall_time": speed}
            for fn in score_fn:
                if isinstance(fn, str):
                    fn = def_scorers[fn]
                scores.update(fn(examples))
            scores = flatten_dict(scores)
            scores_ls.append(scores)

        # and collapse list to dict
        for key in scores.keys():
            scores[key] = [s[key] if key in s else None for s in scores_ls]

        scores["k"] = list(range(k))

        return pd.DataFrame(scores)

    for i, aug in enumerate(augmenters):
        scores_ = __score(aug)
        scores = pd.concat([scores, scores_]) if i != 0 else scores_
    return scores
