# to download the danlp and nerda you will have to set up a certificate:
import ssl
from typing import Iterable, List

from NERDA.precooked import DA_BERT_ML
from spacy.lang.da import Danish
from spacy.tokens import Doc
from spacy.training import Example

from .apply_fn_utils import add_iob, no_misc_getter

ssl._create_default_https_context = ssl._create_unverified_context

model = DA_BERT_ML()
# model.download_network()
model.load_network()
nlp_da = Danish()


def apply_nerda(examples: Iterable[Example], use_spacy: bool = True) -> List[Example]:
    sentences = []
    docs_y = []
    for example in examples:
        # tokenization
        if use_spacy:
            sentences.append([t.text for t in nlp_da(example.reference.text)])
        else:
            from nltk.tokenize import word_tokenize

            sentences.append(word_tokenize(example.reference.text))
        docs_y.append(example.reference)

    # ner
    labels = model.predict(sentences=sentences)

    examples_ = []
    for doc_y, label, words in zip(docs_y, labels, sentences):
        if len(label) < len(words):
            label += ["O"] * (len(words) - len(label))

        doc = Doc(nlp_da.vocab, words=words)
        doc = add_iob(doc, iob=label)
        examples_.append(Example(doc, doc_y))
    return examples_


if __name__ == "__main__":
    import os

    os.chdir("..")
    from dacy.datasets import dane

    test = dane(splits=["test"])
    nlp = Danish()
    examples = apply_nerda(test(nlp), use_spacy=True)

    from spacy.scorer import Scorer

    tok_scores = Scorer.score_tokenization(examples)
    ent_scores = Scorer.score_spans(
        examples=examples,
        attr="ents",
        getter=no_misc_getter,
    )
    pos_scores = Scorer.score_token_attr(examples, "tag")

    from spacy import displacy

    displacy.render(examples[0].y, style="ent")
    displacy.render(examples[0].x, style="ent")
