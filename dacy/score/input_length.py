"""
Contains functions for testing the performance of models on varying input length.
"""
from typing import Callable, List, Union

import pandas as pd

from ..datasets import dane
from .score import score


def n_sents_score(
    n_sents: Union[int, List[int]],
    apply_fn: Callable,
    dataset: str = "dane",
    split: str = "test",
    score_fn: List[Union[str, Callable]] = ["token", "pos", "ents", "dep"],
    verbose: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """scores the performance of a given model on examples of a given number of sentences.

    Args:
        n_sents (Union[int, List[int]]): Number of sentences which the performance should be applied to.
        apply_fn (Callable):  A wrapper function for the model you wish to score. The model should take in a spacy Example and output a tagged version of it.
        dataset (str, optional): Which dataset should this be applied to. Possible options include "dane". Defaults to "dane".
        split (str, optional): Which splits of the dataset should be used. Possible options include "train", "dev", "test", "all".
            Defaults to "test".
        score_fn (List[Union[str, Callable]], optional): A scoring function which takes in a list of examples and return a dictionary of the form {"score_name": score}.
            Four potiential strings are valid. "ents" for measuring the performance of entity spans. "pos" for measuring the performance of pos-tags.
            "token" for measuring the performance of tokenization. "nlp" for measuring the performance of all components in the specified nlp pipeline. Defaults to ["token", "pos", "ents"].
        verbose (bool, optional): Toggles the verbosity of the function. Defualts to True
        kwargs (dict): arguments to be passed to dataset or the score function.

    Returns:
        pandas.DataFrame: returns a pandas dataframe containing the performance metrics.
    """

    dataset_fn = {"dane": dane}
    if isinstance(n_sents, int):
        n_sents = [n_sents]

    k = kwargs["k"] if "k" in kwargs else 1

    for i, n in enumerate(n_sents):
        if verbose is True:
            print(f"[INFO] Calculating score using {n} sentences")
        corpus = dataset_fn[dataset](splits=split, n_sents=n, **kwargs)
        scores_ = score(corpus, apply_fn=apply_fn, score_fn=score_fn, **kwargs)
        scores = pd.concat([scores, scores_]) if i != 0 else scores_
    return scores
