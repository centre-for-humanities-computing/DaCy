### pip install daluke==0.0.5
from typing import Iterable, List

from daluke import AutoNERDaLUKE, predict_ner
from spacy.lang.da import Danish
from spacy.tokens import Doc
from spacy.training import Example

from .apply_fn_utils import add_iob, no_misc_getter

# This also downloads daluke model (first time)
daluke = AutoNERDaLUKE()

nlp_da = Danish()


def apply_daluke(
    examples: Iterable[Example],
    use_spacy: bool = True,
    batch_size: int = 16,
) -> List[Example]:
    docs_y, sentences = list(), list()
    for example in examples:
        # Tokenization using spacy or nltk
        if use_spacy:
            sentences.append([t.text for t in nlp_da(example.reference.text)])
        else:
            from nltk.tokenize import word_tokenize

            sentences.append(word_tokenize(example.reference.text))
        docs_y.append(example.reference)
    # NER using daluke
    # join `should` not give size issues, as this string is again crudely split in DaLUKE API
    predictions = predict_ner(
        [" ".join(sent) for sent in sentences],
        daluke,
        batch_size=batch_size,
    )
    out_examples = list()
    for doc_y, pred, words in zip(docs_y, predictions, sentences):
        doc = add_iob(Doc(nlp_da.vocab, words=words), iob=pred)
        out_examples.append(Example(doc, doc_y))
    return out_examples


if __name__ == "__main__":
    import os

    os.chdir("..")
    from dacy.datasets import dane

    test = dane(splits=["test"])
    nlp = Danish()
    examples = apply_daluke(test(nlp))

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
    breakpoint()
