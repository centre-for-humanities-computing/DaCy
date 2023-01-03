import spacy

import dacy  # noqa


def test_add_hate_speech_detection():
    nlp = spacy.blank("da")
    nlp.add_pipe("dacy/hatespeech_detection")
    texts = ["senile gamle idiot", "hej har du haft en god dag"]
    actual = ["offensive", "not offensive"]
    docs = list(nlp.pipe(texts))
    for d, a in zip(docs, actual):
        if a is None:
            assert d._.is_offensive is None
        else:
            assert d._.is_offensive == a


def test_add_bertemotion_emo():
    nlp = spacy.blank("da")
    nlp.add_pipe("dacy/hatespeech_detection")
    nlp.add_pipe("dacy/hatespeech_classification")
    doc = nlp("senile gamle idiot")
    assert doc._.hate_speech_type == "sprogbrug"
