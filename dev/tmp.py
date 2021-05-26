import time

from flair.data import Sentence, Token
from utils import print_speed_performance, f1_report

from danlp.datasets import DDT
from danlp.models import load_spacy_model, load_flair_ner_model, load_bert_ner_model


def is_misc(ent: str):
    if len(ent) < 4:
        return False
    return ent[-4:] == 'MISC'


def remove_miscs(se: list):
    return [
        [entity if not is_misc(entity) else 'O' for entity in entities]
        for entities in se
    ]


# Load the DaNE data
_, _, test = DDT().load_as_simple_ner(predefined_splits=True)
sentences_tokens, sentences_entities = test

# Replace MISC with O for fair comparisons
sentences_entities = remove_miscs(sentences_entities)

num_sentences = len(sentences_tokens)
num_tokens = sum([len(s) for s in sentences_tokens])

bert = load_bert_ner_model()

start = time.time()

predictions = []
for i, sentence in enumerate(sentences_tokens):
    _, pred_ents = bert.predict(sentence)
    predictions.append(pred_ents)

assert len(predictions) == num_sentences

print(f1_report(sentences_entities, remove_miscs(predictions), bio=True))

res = []
d  = set()
def trans(i):
    d.add(i)
    if i == "PER" or i == "I-PER" or i == "B-PER":
        return 1
    else:
        return 0
for s, ss in zip(sentences_entities, remove_miscs(predictions)):
    res += list(zip([trans(i) for i in ss], [trans(i) for i in s]))

import sklearn
import numpy as np
arr = np.array(res)
sklearn.metrics.f1_score(arr.T[0], arr.T[1], pos_label=1)


sentences_tokens[0]
examples[0].reference
examples = [apply_model(e) for e in corpus(nlp)]