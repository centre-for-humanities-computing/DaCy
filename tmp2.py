
from dacy.datasets import dane
import spacy 
test = dane(splits=["test"])

nlp = spacy.load("da_core_news_sm")

from dacy.score import score

out = score(corpus = test, apply_fn = nlp , score_fn = ["dep"], nlp = nlp)

out
