import os
from pathlib import Path

import spacy
from danlp.datasets import DDT
from danlp.models import load_bert_ner_model
from spacy import displacy
from spacy.cli.convert import conllu_to_docs
from spacy.scorer import Scorer
from spacy.tokens import Doc, Span
from spacy.training import Corpus, Example
from spacy.training.augment import create_lower_casing_augmenter

ddt = DDT()
train, dev, test = ddt.load_as_conllu(predefined_splits=True)

path = "assets/dane/"
Path(path).mkdir(parents=True, exist_ok=True)
save_path = os.path.join(path, "dane_test.conllu")
with open(save_path, "w") as f:
    test.write(f)

Path("corpus/dane").mkdir(parents=True, exist_ok=True)
os.system(
    "python -m spacy convert assets/dane/dane_test.conllu corpus/dane --converter conllu --merge-subtokens -n 1"
)

corpus = Corpus("corpus/dane/dane_test.spacy")

# Checking the small danish model
nlp = spacy.load("da_core_news_sm")


def apply_model(example):
    example.predicted = nlp(example.predicted.text)
    return example


examples = [apply_model(e) for e in corpus(nlp)]

scorer = Scorer()
scores = scorer.score_spans(examples, "ents")

corpus_lower = Corpus(
    "corpus/dane/dane_test.spacy", augmenter=create_lower_casing_augmenter(level=1)
)

words = ["hello", "world", "!"]
spaces = [True, False, False]
predicted = Doc(nlp.vocab, words=words, spaces=spaces)
example = Example(predicted, predicted)

examples_lower = [apply_model(e) for e in corpus_lower(nlp)]
scores_lower = scorer.score_spans(examples_lower, "ents")

# displacy.render(examples[0].predicted, style="ent")
# displacy.render(examples_lower[0].predicted, style="ent")
# displacy.render(examples[0].reference, style="ent")

# Checking DaNLPs NER bert
from transformers import pipeline

bert = load_bert_ner_model()

example = examples[28]


def apply_bert_model(example, bert_model):
    doc = example.predicted
    # uses spacy tokenization
    tokens, labels = bert_model.predict([t.text for t in example.predicted])
    ent = []
    for i, t in enumerate(zip(doc, labels)):
        token, label = t

        # turn OOB labels into spans
        if label == "O":
            continue
        iob, ent_type = label.split("-")
        if (i - 1 >= 0 and iob == "I" and labels[i - 1] == "O") or (
            i == 0 and iob == "I"
        ):
            iob = "B"
        if iob == "B":
            start = i
        if i + 1 >= len(labels) or labels[i + 1].split("-")[0] != "I":
            ent.append(Span(doc, start, i + 1, label=ent_type))
    doc.set_ents(ent)
    example.predicted = doc
    return example



text = "det her er en tekst med et navn der er Lasse"
bert.predict(text.split())
bert.predict("det her en test")
bert.predict("det her en test".split(), IOBformat=False)

tekst_tokenized = ['Han', 'hedder', 'Anders', 'And', 'Andersen', 'og', 'bor', 'i', 'Århus', 'C']
bert.predict(tekst_tokenized, IOBformat=False)

apply_bert_model(example, bert)

apply_

from dacy.datasets import dane
test = dane(splits=["test"])

examples_bert = [apply_bert_model(e, bert) for i, e in enumerate(test(nlp))]
# Examine predictions
# displacy.render(examples_bert[28].reference, style="ent")
# displacy.render(examples_bert[28].predicted, style="ent")
scores_bert = scorer.score_spans(examples_bert, "ents", allow_overlap=True)
scores_bert

examples_bert_lower = [apply_bert_model(e, i) for i, e in enumerate(corpus_lower(nlp))]
scores_bert_lower = scorer.score_spans(examples_bert_lower, "ents", allow_overlap=True)
scores_bert_lower
