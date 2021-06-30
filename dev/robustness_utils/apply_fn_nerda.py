from typing import Iterable, List
from spacy.tokens import Span, Doc
from spacy.training import Example
from spacy.lang.da import Danish


from dev.apply_fn_utils import add_iob, no_misc_getter

from NERDA.precooked import DA_BERT_ML

model = DA_BERT_ML()
#model.download_network()
model.load_network()
nlp = Danish()

from nltk.tokenize import word_tokenize

def apply_nerda(examples: Iterable[Example], use_spacy: bool=False) -> List[Example]:

    sentences = []
    docs_y = []
    for example in examples:
        # tokenization
        if use_spacy:
            sentences.append([t.text for t in nlp(example.reference.text)])
        else:
            sentences.append(word_tokenize(example.reference.text))
        docs_y.append(example.reference)

    # ner
    labels = model.predict(sentences=sentences)

    examples_ = []
    for doc_y, label, words in zip(docs_y, labels, sentences):
        doc = Doc(nlp.vocab, words=words, ents=label)
        examples_.append(Example(doc, doc_y))
    return examples_


if __name__ == "__main__":
    import os

    os.chdir("..")
    from dacy.datasets import dane
    from spacy.lang.da import Danish

    test = dane(splits=["test"])
    nlp = Danish()
    examples = apply_nerda(test(nlp), use_spacy=True)

    from spacy.scorer import Scorer

    tok_scores = Scorer.score_tokenization(examples)
    ent_scores = Scorer.score_spans(
        examples=examples, attr="ents", getter=no_misc_getter
    )
    pos_scores = Scorer.score_token_attr(examples, "tag")

    from spacy import displacy

    displacy.render(examples[0].y, style="ent")
    displacy.render(examples[0].x, style="ent")
