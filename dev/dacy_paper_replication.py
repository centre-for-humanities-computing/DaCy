# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Augmentation
# This notebook recreates Table X from the paper XX and illustrates how to use the augmenters and scoring functions included in DaCy

# %%
# ^should be true if not:
#!pip install spacy[cuda102]


# %%
# import os  # assuming we are located in dacy github repo

# os.chdir("..")
import sys
sys.path
sys.path.append('/home/kenneth/github/DaCy')
import dacy

# %%
# install dacy if not already installed
#!pip install -r requirements.txt # assumed version 1.0.0 of dacy
# or using
#!pip install dacy

# download relevant spacy models
#!python -m spacy download da_core_news_sm
#!python -m spacy download da_core_news_md
#!python -m spacy download da_core_news_lg

# download danlp dependencies
#!pip install danlp==0.0.11
#!pip install transformers==3.5.1 --no-deps # for DaNLP
#!pip install NERDA

# %% [markdown]
# # The dataset: DaNE
# Start off by loading the test set of the DaNE dataset.

# %%
from dacy.datasets import dane

test = dane(splits=["test"])

# %% [markdown]
# # Augmenters
#
# Create a list of augmenters we wish to apply to our model.

# %%
from spacy.training.augment import create_lower_casing_augmenter, dont_augment
from dacy.augmenters import (
    create_pers_augmenter,
    create_keyboard_augmenter,
    create_æøå_augmenter,
)
from dacy.datasets import danish_names, muslim_names, female_names, male_names

# randomly augment names
dk_name_dict = danish_names()
muslim_name_dict = muslim_names()
f_name_dict = female_names()
m_name_dict = male_names()

dk_aug = create_pers_augmenter(dk_name_dict, force_size=True, keep_name=False)
muslim_aug = create_pers_augmenter(muslim_name_dict, force_size=True, keep_name=False)
f_aug = create_pers_augmenter(dk_name_dict, force_size=True, keep_name=False)
m_aug = create_pers_augmenter(muslim_name_dict, force_size=True, keep_name=False)
punct_aug = create_pers_augmenter(
    muslim_name_dict, force_size=False, keep_name=True, patterns=["abbpunct"]
)


# randomly change 5%/15% of characters to a neighbouring key
keyboard_aug_05 = create_keyboard_augmenter(
    doc_level=1, char_level=0.05, keyboard="QWERTY_DA"
)
keyboard_aug_15 = create_keyboard_augmenter(
    doc_level=1, char_level=0.15, keyboard="QWERTY_DA"
)

# Change æ=ae, ø=oe, å=aa
æøå_aug = create_æøå_augmenter(doc_level=1, char_level=1)

# lower case text
lower_case_aug = create_lower_casing_augmenter(level=1)

n = 20
# augmenter   name               n rep
augmenters = [
    (dont_augment, "No augmentation", 1),
    (keyboard_aug_05, "Keyboard augmentation 5%", n),
    (keyboard_aug_15, "Keyboard augmentation 15%", n),
    (æøå_aug, "Æøå augmentation", 1),
    (lower_case_aug, "Lowercase augmentation", 1),
    (dk_aug, "Danish names augmentation", n),
    (muslim_aug, "Muslim names augmentation", n),
    (f_aug, "Female names augmentation", n),
    (m_aug, "Male names augmentation", n),
    (punct_aug, "Abbreviated names augmentation", 1),
]

# %% [markdown]
# # Apply functions
# Defining application functions for necessary models. No need to create one for SpaCy pipelines.

# %%
from spacy.tokens import Span


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


def apply_nerda_model(example, bert_model):
    doc = example.predicted
    # uses spacy tokenization
    labels = bert_model.predict(
        [[t.text for t in example.predicted]]
    )  # nerda requires it to be list of list of tokens
    labels = labels[0]
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
    ### DaNLP's BERT model requires transformers==3.5.1 (install with pip install transformers==3.5.1 --no-deps)


# %% [markdown]
# # Models
# A list of models to apply. To save memory the models are only loaded in one at a time.

# %%
from danlp.models import load_bert_ner_model
from NERDA.precooked import DA_BERT_ML

from NERDA.precooked import DA_BERT_ML

model = DA_BERT_ML()
#model.download_network()
model.load_network()

model_dict = {
    # "spacy_small" : "da_core_news_sm",
    # "spacy_medium": "da_core_news_md",
    # "spacy_large" : "da_core_news_lg",
    #"dacy_small": "da_dacy_small_tft-0.0.0",
    # "dacy_medium" : "da_dacy_medium_tft-0.0.0",
    # "dacy_large" : "da_dacy_large_tft-0.0.0",
    # "danlp_bert" : load_bert_ner_model,
    "nerda_bert": model,
}


# %%
# to download the danlp you will have to set up a certificate:
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

# %% [markdown]
# # Performance

# %%
from pathlib import Path

Path("robustness").mkdir(parents=True, exist_ok=True)


# %%
from functools import partial
import pandas as pd
import spacy

from dacy.score import score, n_sents_score


for mdl in model_dict:
    print(f"[INFO]: Scoring model '{mdl}' using DaCy")

    # load model
    if "dacy" in mdl:
        apply_fn = dacy.load(model_dict[mdl])
    elif "spacy" in mdl:
        apply_fn = spacy.load(model_dict[mdl])
    elif mdl == "danlp_bert":
        bert = model_dict[mdl]()
        apply_fn = partial(apply_bert_model, bert_model=bert)
    else:
        apply_fn = partial(apply_nerda_model, bert_model=model)

    i = 0
    scores = []
    for aug, nam, k in augmenters:
        print(f"\t Running augmenter: {nam}")

        scores_ = score(corpus=test, apply_fn=apply_fn, augmenters=aug, k=k)
        scores_["model"] = mdl
        scores_["augmenter"] = nam
        scores_["i"] = i
        print(scores_)
        scores.append(scores_)

        i += 1

    for n in [5, 10]:
        scores_ = n_sents_score(n_sents=n, apply_fn=apply_fn)
        scores_["model"] = mdl
        scores_["augmenter"] = f"Input size augmentation {n} sentences"
        scores_["i"] = i + 1
        scores.append(scores_)
    scores = pd.concat(scores)

    scores.to_csv(f"robustness/{mdl}_augmentation_performance.csv")
