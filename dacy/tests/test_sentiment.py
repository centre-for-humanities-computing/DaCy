from dacy.sentiment import (
    add_berttone_subjectivity,
    add_berttone_polarity,
    add_bertemotion_laden,
    add_bertemotion_emo,
    add_senda,
)
import dacy
import spacy


def test_add_berttone_subjectivity():
    nlp = spacy.blank("en")
    nlp = add_berttone_subjectivity(
        nlp, open_unverified_connection=True, force_extension=True
    )
    texts = [
        "Analysen viser, at økonomien bliver forfærdelig dårlig",
        "Jeg tror alligevel, det bliver godt",
    ]
    actual = ["objective", "subjective"]
    docs = nlp.pipe(texts)
    for d, a in zip(docs, actual):
        assert d._.subjectivity == a


def test_add_berttone_polarity():
    nlp = spacy.blank("en")
    nlp = add_berttone_polarity(
        nlp, open_unverified_connection=True, force_extension=True
    )

    texts = [
        "Analysen viser, at økonomien bliver forfærdelig dårlig",
        "Jeg tror alligevel, det bliver godt",
    ]
    docs = nlp.pipe(texts)

    l = [d for d in docs]
    # text 0 should be more negative that text 1
    assert l[0]._.polarity_prop["prop"][0] < l[1]._.polarity_prop["prop"][0]


def test_add_bertemotion_laden():
    nlp = spacy.blank("da")
    nlp = add_bertemotion_laden(
        nlp, open_unverified_connection=True, force_extension=True
    )


def test_add_bertemotion_emo():
    nlp = spacy.blank("da")
    nlp = add_bertemotion_emo(
        nlp, open_unverified_connection=True, force_extension=True
    )
    doc = nlp("Har i set at Tesla har landet en raket på månen? Det er vildt!!")
    assert doc._.emotion == "Overasket/Målløs"


def test_add_senda():
    nlp = spacy.blank("da")
    nlp = add_senda(nlp, force_extension=True)
    doc = nlp("Sikke en dejlig dag det er i dag")
    assert doc._.polarity == "positive"


def test_add_davader():
    # using small danish spacy model as DaCy does not include a lemmatizer (yet)
    try:
        nlp = spacy.load("da_core_news_sm")
    except OSError:
        print(
            "Downloading language model for the spaCy tokenization\n"
            "(don't worry, this will only happen once)"
        )
        from spacy.cli import download

        download("da_core_news_sm")
    nlp = spacy.load("da_core_news_sm")

    from spacy.tokens import Doc
    from dacy.sentiment import da_vader_getter
    from functools import partial

    for func in [da_vader_getter, partial(da_vader_getter, lemmatization=False)]:
        Doc.set_extension("vader_da", getter=func, force=True)

        texts = ["Jeg er så glad", "jeg er rastløs"]

        docs = list(nlp.pipe(texts))

        assert docs[0]._.vader_da["compound"] > 0
        assert docs[1]._.vader_da["neg"] > 0
