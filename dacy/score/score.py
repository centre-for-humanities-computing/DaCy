"""
This includes function for scoring models applied to a SpaCy corpus.
"""

from __future__ import annotations

from copy import copy
from functools import partial
from typing import Callable, List, Optional, Union

import pandas as pd
from spacy.language import Language
from spacy.scorer import Scorer
from spacy.training import Corpus
from spacy.training import dont_augment

from ..utils import flatten_dict


def score(
    corpus: Corpus,
    apply_fn: Union[Callable, Language],
    score_fn: List[Union[Callable, str]] = ["token", "pos", "ents"],
    augmenters: Union[List[Callable], Callable] = [],
    k: int = 1,
    nlp: Optional[Language] = None,
    **kwargs,
) -> pd.DataFrame:
    """scores a models performance on a given corpus with potentially augmentations applied to it.

    Args:
        corpus (Corpus): A spacy Corpus
        apply_fn (Union[Callable, Language]): A wrapper function for the model you wish to score. The model should
            take in a spacy Example and output a tagged version of it. A SpaCy pipeline can be provided as is and a
            wrapper will be created for it.
        score_fn (List[Union[Callable, str]], optional): A scoring function which takes in a list of
            examples and return a dictionary of the form {"score_name": score}.Four potiential
            strings are valid. "ents" for measuring the performance of entity spans."pos" for measuring
            the performance of pos-tags. "token" for measuring the performance of tokenization. "nlp"
            for measuring the performance of all components in the specified nlp pipeline. Defaults to
            ["token", "pos", "ents"].
        augmenters (Union[List[Callable], Callable], optional): A spaCy style augmenter which should be
            applied to the corpus or a list thereof. defaults to [], indicating no augmenters.
        k (int, optional): Number of times it should run the augmentation and test the performance on
            the corpus. Defaults to 1.
        nlp (Optional[Language], optional): A spacy processing pipeline. If None it will use an empty
            Danish pipeline. Defaults to None.

    Returns:
        pandas.DataFrame: returns a pandas dataframe containing the performance metrics.

    Example:
        >>> from spacy.training.augment import create_lower_casing_augmenter
        >>> train, dev, test = dacy.datasets.dane(predefined_splits=True)
        >>> def apply_model(example):
                example.predicted = nlp(example.predicted.text)
                return example
        >>> scores = scores(test, augmenter=create_lower_casing_augmenter(0.5), apply_fn = apply_model)
    """

    def __apply_nlp(example):
        example.predicted = nlp_(example.reference.text)
        return example

    if nlp is None:
        from spacy.lang.da import Danish

        nlp = Danish()

    if callable(augmenters):
        augmenters = [augmenters]
    if len(augmenters) == 0:
        augmenters = [dont_augment]

    if isinstance(apply_fn, Language):
        nlp_ = apply_fn
        apply_fn = __apply_nlp

    scorer = Scorer(nlp)
    def_scorers = {
        "ents": partial(Scorer.score_spans, attr="ents"),
        "pos": partial(Scorer.score_token_attr, attr="pos"),
        "token": Scorer.score_tokenization,
        "nlp": scorer.score,
    }

    def __score(augmenter):

        corpus_ = copy(corpus)
        corpus_.augmenter = augmenter
        scores_ls = []
        for i in range(k):
            examples = [apply_fn(e) for e in corpus_(nlp)]
            scores = {}
            for fn in score_fn:
                if isinstance(fn, str):
                    fn = def_scorers[fn]
                scores.update(fn(examples))
            scores = flatten_dict(scores)
            scores_ls.append(scores)

        # and collapse list to dict
        for key in scores.keys():
            scores[key] = [s[key] for s in scores_ls]

        scores["k"] = list(range(k))

        return pd.DataFrame(scores)

    for i, aug in enumerate(augmenters):
        scores_ = __score(aug)
        scores = pd.concat([scores, scores_]) if i != 0 else scores_
    return scores
