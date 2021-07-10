
import spacy
from spacy.lang.da import Danish
import pandas as pd

import dacy
from dacy.datasets import dane
from dacy.score import score, n_sents_score

from spacy.training.augment import create_lower_casing_augmenter


def test_score():
    nlp = Danish()

    def apply_model(examples):
        e = []
        for example in examples:
            example.predicted = nlp(example.predicted.text)
            e.append(example)
        return e

    test = dane(splits=["test"])
    test.limit = 1

    scores = score(
        corpus=test,
        augmenter=create_lower_casing_augmenter(0.5),
        apply_fn=apply_model,
        k=3,
        score_fn=["ents", "pos", "token"],
    )
    assert isinstance(scores, pd.DataFrame)
    
    # test with nlp as input
    scores_ = score(
        corpus=test,
        augmenter=create_lower_casing_augmenter(0.5),
        apply_fn=nlp,
        k=3,
        score_fn=["ents", "pos", "token"],
    )
    for s in scores_:
        assert s in scores.columns

def test_n_sents_score():
    nlp = Danish()

    def apply_model(examples):
        e = []
        for example in examples:
            example.predicted = nlp(example.predicted.text)
            e.append(example)
        return e

    scores = n_sents_score(
        n_sents=1,
        apply_fn=apply_model,
    )
    assert isinstance(scores, pd.DataFrame)