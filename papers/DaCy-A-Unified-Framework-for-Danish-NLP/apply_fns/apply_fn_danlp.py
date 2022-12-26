### DaNLP's BERT model requires transformers==3.5.1 (install with pip install transformers==3.5.1 --no-deps)

# to download the danlp and nerda you will have to set up a certificate:
import ssl

from danlp.models import load_bert_ner_model
from spacy.lang.da import Danish
from spacy.training import Example

from .apply_fn_utils import add_iob, apply_on_multiple_examples, no_misc_getter

ssl._create_default_https_context = ssl._create_unverified_context

bert_model = load_bert_ner_model()
nlp_da = Danish()


def __apply_bert_model(example: Example) -> Example:
    # uses spacy tokenization
    doc = nlp_da(example.reference.text)
    tokens, labels = bert_model.predict([t.text for t in doc])
    doc = add_iob(doc, labels)
    return Example(doc, example.reference)


apply_danlp_bert = apply_on_multiple_examples(__apply_bert_model)


if __name__ == "__main__":
    import os

    os.chdir("..")
    from dacy.datasets import dane

    test = dane(splits=["test"])
    nlp = Danish()
    examples = apply_danlp_bert(test(nlp))

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
