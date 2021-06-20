from typing import Callable, Union
from spacy.training import Corpus

from spacy.scorer import Scorer
from functools import partial

import spacy

import dacy

from spacy.training.augment import create_lower_casing_augmenter


def test_score():
    nlp = dacy.load("da_dacy_medium_tft-0.0.0")

    def apply_model(example):
        example.predicted = nlp(example.predicted.text)
        return example

    train, dev, test = dacy.datasets.dane(predefined_splits=True)

    scores = dacy.testing.score(
        corpus=test,
        augmenter=create_lower_casing_augmenter(0.5),
        apply_fn=apply_model,
        k=3,
        score_fn=["ents", "pos", "token"],
    )
    print(scores)
    scores.to_df()


def test_n_sents_score():
    nlp = dacy.load("da_dacy_medium_tft-0.0.0")

    def apply_model(example):
        example.predicted = nlp(example.predicted.text)
        return example

    scores = dacy.testing.n_sents_score(
        n_sents=1,
        apply_fn=apply_model,
    )
    print(scores)

def test_Scores():
    scores = dacy.testing.Scores({"acc": [0.999999, 0.988888, 0.7666]})
    assert isinstance(scores.scores, dict)
    assert isinstance(scores.summary("acc"), str)
    scores.to_df()