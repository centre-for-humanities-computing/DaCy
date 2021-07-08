## pip install spacy-stanza

import stanza
import spacy_stanza

import os
os.chdir("..")
from dacy.datasets import dane
from spacy.scorer import Scorer

stanza.download("da")
nlp = spacy_stanza.load_pipeline("da", lang="da")

test = dane(splits=["test"])
examples = list(test(nlp))

Scorer.score_tokenization(examples)
Scorer.score_spans(examples=examples, attr="ents", allow_overlap=True)
Scorer.score_token_attr(examples, "pos")
Scorer.score_token_attr(examples, "pos") #dep 


# rewrite score func
# score_fn to include tag
