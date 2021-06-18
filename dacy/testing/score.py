from typing import Callable, List, Optional, Union
from spacy.training import Corpus

from spacy.scorer import Scorer
from functools import partial
from spacy.language import Language

from pydantic import BaseModel
import numpy as np

from ..utils import flatten_dict

class Scores(BaseModel):
    scores: dict

    def mean(self, score: str):
        s = self.scores[score]
        return np.mean(s)

    def std(self, score: str):
        s = self.scores[score]
        return np.std(s)
    
    def summary(self, score: str):
        std = self.std(score)
        mean = self.mean(score)
        return f"{round(mean, 2)} ({round(std, 2)})"

    def to_df(self):
        import pandas as pd
        return pd.DataFrame(self.scores)

    def __repr_str__(self, join_str: str) -> str:
        return join_str.join(
            repr(v) if a is None else f"{a}={v!r}"
            for a, v in [(k, self.summary(k)) for k in self.scores.keys()]
        )


def score(
    corpus: Corpus,
    augmenter: Optional[Callable],
    apply_fn: Callable,
    score_fn: List[Union[Callable, str]] = ["token", "pos", "ents"],
    nlp: Optional[Language] = None,
    k: int = 1,
) -> Scores:
    """scores a models performance on a given corpus with potentially augmentations applied to it.

    Args:
        corpus (Corpus): A spacy Corpus
        augmenter (Optional[Callable]): A spacy style augmenter which should be applied to the corpus.
        apply_fn (Callable): A wrapper function for the model you wish to score. The model should take in a spacy Example and output a tagged version of it.
        score_fn (List[Union[Callable, str]], optional): A scoring function which takes in a list of examples and return a dictionary of the form {"score_name": score}. 
            Four potiential strings are valid. "ents" for measuring the performance of entity spans. "pos" for measuring the performance of pos-tags.
            "token" for measuring the performance of tokenization. "nlp" for measuring the performance of all components in the specified nlp pipeline. Defaults to ["token", "pos", "ents"].
        nlp (Optional[Language], optional): A spacy processing pipeline. If None it will use an empty Danish pipeline. Defaults to None.
        k (int, optional): Number of times it should run the augmentation and test the performance on the corpus. Defaults to 1.

    Returns:
        Scores: returns a Score dataclass. Which contain the scores dictionary and convenience function for printing and turning it into a dataframe.

    Example:
        >>> from spacy.training.augment import create_lower_casing_augmenter
        >>> train, dev, test = dacy.datasets.dane(predefined_splits=True)
        >>> def apply_model(example):
                example.predicted = nlp(example.predicted.text)
                return example
        >>> scores = scores(test, augmenter=create_lower_casing_augmenter(0.5), apply_model)
        >>> scores.scores # extract dictionary of scores
        >>> scores.to_df() # creates a pandas dataframe of scores 
    """
    # scorer default to the spacy scorer.
    # if none default the spacy
    if nlp is None:
        from spacy.lang.da import Danish

        nlp = Danish()
    scorer = Scorer(nlp)
    def_scorers = {
        "ents": partial(Scorer.score_spans, attr="ents"),
        "pos": partial(Scorer.score_token_attr, attr="pos"),
        "token": Scorer.score_tokenization,
        "nlp": scorer.score,
    }

    if augmenter is not None:
        corpus.augmenter = augmenter

    scores_ls = []
    for i in range(k):
        examples = [apply_fn(e) for e in corpus(nlp)]
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
    return Scores(scores=scores)


