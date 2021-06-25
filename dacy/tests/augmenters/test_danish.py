from dacy.augmenters.danish import create_æøå_augmenter
from spacy.lang.da import Danish

def test_create_æøå_augmenter():
    augmenter = create_æøå_augmenter(doc_level=1, char_level=1)
    nlp = Danish()
    doc = nlp("æøå")
    doc = augmenter(doc)
    assert doc[0] == "aeoeaa"

