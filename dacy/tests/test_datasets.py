
import dacy

from spacy.training import Corpus
from spacy.lang.da import Danish
from spacy.training import Example

def test_dane():
    train, dev, test = dacy.datasets.dane(predefined_splits=True)
    all_ = dacy.datasets.dane(predefined_splits=False)
    for d in [train, dev, test, all_]:
        assert isinstance(d, Corpus)

    nlp = Danish()
    examples = list(test(nlp)) # check if it read as intended
    assert isinstance(examples[0], Example)
