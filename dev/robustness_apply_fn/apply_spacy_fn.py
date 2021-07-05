import spacy

import os
import sys
sys.path.append('/home/kenneth/github/DaCy')
# os.chdir("../..")
from dacy.datasets import dane
from dacy.score import score 
from spacy.lang.da import Danish
import dacy

test = dane(splits=["test"])

nlp = dacy.load(dacy.models()[0])

scores = score(corpus=test, apply_fn=nlp)

print(scores)