import spacy

import dacy  # noqa


def test_add_subjectivity():
    nlp = spacy.blank("da")
    nlp.add_pipe("dacy/subjectivity")
    texts = [
        "Analysen viser, at økonomien bliver forfærdelig dårlig",
        "Jeg tror alligevel, det bliver godt",
        "",
    ]
    actual = ["objective", "subjective", None]
    docs = list(nlp.pipe(texts))
    for d, a in zip(docs, actual):
        if a is None:
            assert d._.subjectivity is None
        else:
            assert d._.subjectivity == a


def test_add_berttone_polarity():
    nlp = spacy.blank("da")
    nlp.add_pipe("dacy/polarity")

    texts = [
        "Analysen viser, at økonomien bliver forfærdelig dårlig",
        "Jeg tror alligevel, det bliver godt",
    ]
    docs = list(nlp.pipe(texts))

    # text 0 should be more negative that text 1
    assert docs[0]._.polarity_prob["prob"][0] < docs[1]._.polarity_prob["prob"][0]


def test_add_bertemotion_laden():
    nlp = spacy.blank("da")
    nlp.add_pipe("dacy/emotionally_laden")
    doc = nlp("Der er et træ i haven")
    assert doc._.emotionally_laden == "no emotion"


def test_add_bertemotion_emo():
    nlp = spacy.blank("da")
    nlp.add_pipe("dacy/emotionally_laden")
    nlp.add_pipe("dacy/emotion")
    doc = nlp("Har i set at Tesla har landet en raket på månen? Det er vildt!!")
    assert doc._.emotion == "overasket/målløs"
