"""
Contains functions for testing the performance of models on varying input length.
"""
from functools import partial
from typing import Callable, Optional, List, Union
from .score import score, Scores
from ..datasets import dane


def n_sents_score(
    n_sents: Union[int, List[int]],
    apply_fn: Callable,
    dataset: str = "dane",
    split: str = "test",
    score_fn: List[Union[str, Callable]] = ["token", "pos", "ents"],
    verbose: bool=True,
    **kwargs
) -> Scores:
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
        Scores: returns a Score dataclass. Which contain the scores dictionary and convenience function for printing and turning it into a dataframe.
    """
    # score_fn:  A scoring function which takes in a list of examples and return a dictionary of the form {"score_name": score}.
    # dataset: currently only "dane"
    # kwargs = argumetns to be passed to score

    dataset_fn = {"dane": dane}
    if isinstance(n_sents, int):
        n_sents = [n_sents]

    k = kwargs["k"] if "k" in kwargs else 1

    for i, n in enumerate(n_sents):
        if verbose is True:
            print(f"[INFO] Calculating score using {n_sents} sentences")
        corpus = dataset_fn[dataset](splits=split, n_sents=n, **kwargs)
        scores_ = score(corpus, apply_fn=apply_fn, score_fn=score_fn, **kwargs)
        scores_.scores["n_sents"] = [n]*k
        scores = scores + scores_ if i != 0 else scores_
    return scores