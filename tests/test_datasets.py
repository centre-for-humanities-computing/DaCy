from spacy.lang.da import Danish
from spacy.training import Corpus, Example

import dacy
from dacy.datasets import dane, female_names, male_names, muslim_names

test = dane(splits=["test"])


def test_dane():
    train, dev, test = dane(open_unverified_connection=True)
    all_ = dacy.datasets.dane(splits=["all"])
    for d in [train, dev, test, all_]:
        assert isinstance(d, Corpus)

    nlp = Danish()
    examples = list(test(nlp))  # check if it read as intended
    assert isinstance(examples[0], Example)

    all_ = dacy.datasets.dane(splits=["all"])


def test_names():
    for names in [muslim_names(), female_names(), male_names()]:
        assert isinstance(names["first_name"], list)
        assert len(names["first_name"]) > 0
        assert isinstance(names["last_name"], list)
        assert len(names["last_name"]) > 0
