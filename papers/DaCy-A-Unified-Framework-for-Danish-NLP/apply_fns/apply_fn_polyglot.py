### pip install polyglot morfessor==2.0.6 pycld2==0.41 pyicu
### polyglot download pos2.da

from polyglot.tag import NEChunker, POSTagger
from polyglot.text import Text, WordList
from spacy.lang.da import Danish
from spacy.tokens import Doc
from spacy.training import Example

from .apply_fn_utils import add_iob, apply_on_multiple_examples, no_misc_getter

ne_chunker = NEChunker(lang="da")
pos_tagger = POSTagger(lang="da")


nlp_da = Danish()


def __apply_polyglot(example: Example, use_spacy: bool = False) -> Example:
    # tokenization
    if use_spacy:
        words = WordList(
            [t.text for t in nlp_da(example.reference.text)],
            language="da",
        )
    else:
        text = Text(example.reference.text, hint_language_code="da")
        words = text.words
    # ner
    iob = [iob for token, iob in ne_chunker.annotate(words)]
    # pos-tagging
    tags = [tag for t, tag in pos_tagger.annotate(words)]

    words = [t.string for t in words]
    doc = Doc(nlp_da.vocab, words=words, tags=tags, pos=tags)
    doc = add_iob(doc, iob)
    return Example(doc, example.y)


apply_polyglot = apply_on_multiple_examples(__apply_polyglot)

if __name__ == "__main__":
    import os

    os.chdir("..")
    from dacy.datasets import dane

    test = dane(splits=["test"])
    nlp = Danish()
    examples = apply_polyglot(test(nlp), use_spacy=False)
    examples_spacy = apply_polyglot(test(nlp), use_spacy=True)

    from spacy.scorer import Scorer

    tok_scores = Scorer.score_tokenization(examples)
    ent_scores = Scorer.score_spans(
        examples=examples,
        attr="ents",
        getter=no_misc_getter,
    )
    ent_scores_spacy = Scorer.score_spans(
        examples=examples_spacy,
        attr="ents",
        getter=no_misc_getter,
    )
    pos_scores = Scorer.score_token_attr(examples, "tag")

    from spacy import displacy

    displacy.render(examples[0].y, style="ent")
    displacy.render(examples[0].x, style="ent")
