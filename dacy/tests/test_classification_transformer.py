import os
from unicodedata import decimal

from dacy.wrapped_models import (
    add_berttone_subjectivity,
    add_berttone_polarity,
    add_bertemotion_laden,
    add_bertemotion_emo,
)
import spacy

nlp = spacy.blank("en")
nlp = add_berttone_subjectivity(nlp)
nlp = add_berttone_polarity(nlp)
nlp = add_bertemotion_laden(nlp)
nlp = add_bertemotion_emo(nlp)

from dacy.subclasses import ClassificationTransformer, install_classification_extensions

import dacy

from danlp.download import download_model as danlp_download
from danlp.download import _unzip_process_func
from danlp.download import DEFAULT_CACHE_DIR as DANLP_DIR

path_sub = danlp_download(
    "bert.subjective", DANLP_DIR, process_func=_unzip_process_func, verbose=True
)
path_sub = os.path.join(path_sub, "bert.sub.v0.0.1")

nlp = spacy.blank("en")
config = {
    "doc_extention_attribute": "test",
    "model": {
        # "@architectures": "spacy-transformers.TransformerModel.v1",
        "@architectures": "dacy.ClassificationTransformerModel.v1",
        "name": path_sub,
    },
}

transformer = nlp.add_pipe("classification_transformer", config=config)
transformer.model.initialize()

install_classification_extensions(doc_extention="test")

docs = nlp.pipe(
    [
        "Analysen viser, at økonomien bliver forfærdelig dårlig",
        "Jeg tror alligvel, det bliver godt",
    ]
)
for doc in docs:
    print(doc._.subjectivity)
    print(doc._.subjectivity_prop)
    print(doc._.polarity)
    print(doc._.polarity_prop)
doc._.test
doc._.subjectivity

import numpy as np


def softmax(x):
    return np.exp(x) / sum(np.exp(x))


prop = softmax(x=doc._.clf_trf_data.tensors[0][0]).round(decimals=3)

type(transformer.model.layers[0].shims[0]._model)

# path_sub = danlp_download(
#     "bert.subjective", DANLP_DIR, process_func=_unzip_process_func, verbose=True
# )
# path_sub = os.path.join(path_sub, "bert.sub.v0.0.1")

# from transformers import BertTokenizer, BertForSequenceClassification, AutoModel, AutoModelForSequenceClassification


download_name = "bert.polarity"
subpath = "bert.pol.v0.0.1"
doc_extention = "berttone_pol_trf_data"
model_name = "berttone_pol"
category = "polarity"
labels = ["positive", "neutral", "negative"]

path_sub = danlp_download(
    "bert.polarity", DANLP_DIR, process_func=_unzip_process_func, verbose=True
)
path_sub = os.path.join(path_sub, subpath)
from transformers import AutoModelForSequenceClassification, BertTokenizer

tokenizer_sub = BertTokenizer.from_pretrained(path_sub)
# classes_sub = ["objective", "subjective"]
berttone = AutoModelForSequenceClassification.from_pretrained(path_sub, num_labels=3)
input_ = tokenizer_sub("Jeg er så glad i dag", return_tensors="pt")
out = berttone.forward(**input_)


# TransformerModel(path_sub, forward)

# import spacy
# nlp = spacy.blank("en")
# config = {
#     "model": {
#         "@architectures": "spacy-transformers.TransformerModel.v1",
#         "name": path_sub,
#     }
# }

# transformer = nlp.add_pipe("transformer", config=config)
# transformer.model.initialize()
# from spacy_transformers.span_getters import get_doc_spans
# model = TransformerModel(path_sub, get_doc_spans, {"use_fast": True})
# model.initialize()
# texts = ["the cat sat on the mat.", "hello world."]
# model.predict([nlp(text) for text in texts])
# doc = nlp("Jeg er så glad i dag")
# doc._.trf_data.tensors[0]
# doc._.trf_data.tensors[1]


# from thinc.api import PyTorchWrapper

# from transformers import AutoModelForSequenceClassification, BertTokenizer
# tokenizer_sub = BertTokenizer.from_pretrained(path_sub)
# input_ = tokenizer_sub("Jeg er så glad i dag", return_tensors='pt')
# # classes_sub = ["objective", "subjective"]
# berttone = AutoModelForSequenceClassification.from_pretrained(
#     path_sub, num_labels=2)

# wrapped_berttone = PyTorchWrapper(berttone)
# wrapped_berttone(*input_)