import spacy

from spacy.training import Corpus
from spacy.scorer import Scorer

import dacy
from dacy.augmenters.common_utils import make_danish_name_dict, make_muslim_name_dict
from dacy.augmenters import create_name_augmenter

# Checking the small danish model
nlp = spacy.load("da_core_news_sm")
nlp = dacy.load("da_dacy_small_tft-0.0.0")


def apply_model(example):
    example.predicted = nlp(example.predicted.text)
    return example


ent_dict_muslim = make_muslim_name_dict()
ent_dict_danish = make_danish_name_dict()

corpus_muslim = Corpus(
    "corpus/dane/dane_test.spacy",
    augmenter=create_name_augmenter(ent_dict_muslim, keep_name=False),
)
corpus_danish = Corpus(
    "corpus/dane/dane_test.spacy",
    augmenter=create_name_augmenter(ent_dict_danish, keep_name=False),
)
corpus_raw = Corpus("corpus/dane/dane_test.spacy")

corpus_abb = Corpus(
    "corpus/dane/dane_test.spacy",
    augmenter=create_name_augmenter(
        ent_dict_danish, patterns=["abbpunct"], keep_name=True, force_size=False
    ),
)


examples_muslim = [apply_model(e) for e in corpus_muslim(nlp)]
examples_danish = [apply_model(e) for e in corpus_danish(nlp)]
examples_raw = [apply_model(e) for e in corpus_raw(nlp)]
examples_abb = [apply_model(e) for e in corpus_abb(nlp)]


scorer = Scorer()
scores_muslim = scorer.score_spans(examples_muslim, "ents")
scores_danish = scorer.score_spans(examples_danish, "ents")
scores_raw = scorer.score_spans(examples_raw, "ents")
scores_abb = scorer.score_spans(examples_abb, "ents")