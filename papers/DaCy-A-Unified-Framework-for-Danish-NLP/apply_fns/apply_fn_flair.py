# pip install flair==0.4.5

from typing import Iterable, List

from danlp.models import load_flair_ner_model, load_flair_pos_model
from flair.data import Sentence, Token
from spacy.lang.da import Danish
from spacy.tokens import Doc
from spacy.training import Example

tagger_ner = load_flair_ner_model()
tagger_pos = load_flair_pos_model()
nlp_da = Danish()


def apply_flair(examples: Iterable[Example], use_spacy=True) -> List[Example]:
    flair_sentences = []
    docs_y = []
    for example in examples:
        # tokenization
        if use_spacy:
            sent = Sentence()
            [sent.add_token(Token(t.text)) for t in nlp_da(example.reference.text)]
        else:
            sent = Sentence(example.reference.text)

        flair_sentences.append(sent)

        docs_y.append(example.reference)

    tagger_ner.predict(flair_sentences, verbose=True)
    tagger_pos.predict(flair_sentences, verbose=True)

    examples_ = []
    for doc_y, f_sent in zip(docs_y, flair_sentences):
        text, iob, upos, ws = zip(
            *[
                (
                    tok.text,
                    tok.tags["ner"].value,
                    tok.tags["upos"].value,
                    tok.whitespace_after,
                )
                for tok in f_sent
            ]
        )
        doc = Doc(nlp_da.vocab, words=text, spaces=ws, tags=upos, ents=iob)
        examples_.append(Example(doc, doc_y))
    return examples_


if __name__ == "__main__":
    import os

    os.chdir("..")
    from dacy.datasets import dane

    test = dane(splits=["test"])
    examples = apply_flair(test(nlp_da), use_spacy=False)
    examples_spacy = apply_flair(test(nlp_da), use_spacy=True)

    from spacy import displacy

    displacy.render(examples[5].y, style="ent")
    displacy.render(examples[5].x, style="ent")

    from spacy.scorer import Scorer

    from .apply_fn_utils import no_misc_getter

    tok_scores = Scorer.score_tokenization(examples)

    ent_scores = Scorer.score_spans(
        examples=examples,
        attr="ents",
        allow_overlap=True,
        getter=no_misc_getter,
    )
    ent_scores_spacy = Scorer.score_spans(
        examples=examples_spacy,
        attr="ents",
        allow_overlap=True,
        getter=no_misc_getter,
    )

    pos_scores = Scorer.score_token_attr(examples, "tag")
    pos_scores = Scorer.score_token_attr(examples, "tag")
