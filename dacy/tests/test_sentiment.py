
import dacy
import spacy



def test_add_subjectivity():
    nlp = spacy.blank("da")
    nlp.add_pipe("subjectivity")
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
    nlp.add_pipe("polarity")

    texts = [
        "Analysen viser, at økonomien bliver forfærdelig dårlig",
        "Jeg tror alligevel, det bliver godt",
    ]
    docs = list(nlp.pipe(texts))

    # text 0 should be more negative that text 1
    assert docs[0]._.polarity_prob["prob"][0] < docs[1]._.polarity_prob["prob"][0]


def test_add_bertemotion_laden():
    nlp = spacy.blank("da")
    nlp.add_pipe("emotionally_laden")
    doc = nlp('Der er et træ i haven')
    assert doc._.emotionally_laden == "no emotion"


def test_add_bertemotion_emo():
    nlp = spacy.blank("da")
    nlp.add_pipe("emotion")
    doc = nlp("Har i set at Tesla har landet en raket på månen? Det er vildt!!")
    assert doc._.emotion == "overasket/målløs"


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
