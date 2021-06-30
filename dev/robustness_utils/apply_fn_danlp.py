
from spacy.tokens import Span, Doc
from spacy.training import Example
from spacy.lang.da import Danish

from danlp.models import load_bert_ner_model

from .apply_fn_utils import apply_on_multiple_examples, add_iob, no_misc_getter

bert_model = load_bert_ner_model()
nlp = Danish()

def __apply_bert_model(example: Example) -> Example:
    # uses spacy tokenization
    doc = nlp(example.reference.text)
    tokens, labels = bert_model.predict([t.text for t in doc])
    doc = add_iob(doc, labels)
    return Example(doc, example.reference)

apply_danlp_bert = apply_on_multiple_examples(__apply_bert_model)


if __name__ == "__main__":
    import os

    os.chdir("..")
    from dacy.datasets import dane
    from spacy.lang.da import Danish

    test = dane(splits=["test"])
    nlp = Danish()
    examples = apply_danlp_bert(test(nlp))

    from spacy.scorer import Scorer

    tok_scores = Scorer.score_tokenization(examples)
    ent_scores = Scorer.score_spans(
        examples=examples, attr="ents", getter=no_misc_getter
    )
    pos_scores = Scorer.score_token_attr(examples, "tag")

    from spacy import displacy

    displacy.render(examples[0].y, style="ent")
    displacy.render(examples[0].x, style="ent")
