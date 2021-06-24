from typing import Callable, Union
from spacy.training import Corpus

from spacy.scorer import Scorer
from functools import partial

import spacy

import dacy

from spacy.training.augment import create_lower_casing_augmenter


nlp = spacy.load("da_core_news_sm")


def apply_model(example):
    example.predicted = nlp(example.predicted.text)
    return example


# train, dev, test = dacy.datasets.dane(predefined_splits=True)


# scores = dacy.testing.score(
#     corpus=test, augmenter=create_lower_casing_augmenter(0.5), apply_fn=apply_model, k=3, score_fn = ["ents"]
# )
# print(scores)

# scores.to_df()

scores = dacy.testing.n_sents_score(n_sents=list(range(1,11)),
    apply_fn= apply_model)

scores.to_df()
print(scores)