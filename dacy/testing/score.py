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
    augmenter: Callable,
    apply_fn=Callable,
    score_fn: List[Union[Callable, str]] = ["token", "pos", "ents"],
    nlp: Optional[Language] = None,
    k: int = 1,
):
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


