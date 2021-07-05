# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Augmentation
# This notebook recreates Table X from the paper XX and illustrates how to use the augmenters and scoring functions included in DaCy

# %%
# ^should be true if not:
#!pip install spacy[cuda102]


# %%
# import os # assuming we are located in dacy github repo
# os.chdir("..")


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
#!pip install gensim==3.8.1 # also danlp
#!pip install NERDA
#!pip install spacy-stanza
#!pip install flair==0.4.5
#!pip install torch==1.7.1 # for flair

#!pip install polyglot # you will need to install polyglot dependencies as well
#!polyglot download pos2.da

# %% [markdown]
# # The dataset: DaNE
# Start off by loading the test set of the DaNE dataset.

import sys

sys.path.append("/home/kenneth/github/DaCy")

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
    create_spacing_augmenter,
)
from dacy.datasets import danish_names, muslim_names, female_names, male_names

# randomly augment names
dk_name_dict = danish_names()
muslim_name_dict = muslim_names()
f_name_dict = female_names()
m_name_dict = male_names()

dk_aug = create_pers_augmenter(
    dk_name_dict,
    force_pattern_size=True,
    keep_name=False,
    patterns=["fn", "fn,ln", "fn,ln,ln"],
)
muslim_aug = create_pers_augmenter(
    muslim_name_dict,
    force_pattern_size=True,
    keep_name=False,
    patterns=["fn", "fn,ln", "fn,ln,ln"],
)
f_aug = create_pers_augmenter(
    dk_name_dict,
    force_pattern_size=True,
    keep_name=False,
    patterns=["fn", "fn,ln", "fn,ln,ln"],
)
m_aug = create_pers_augmenter(
    muslim_name_dict,
    force_pattern_size=True,
    keep_name=False,
    patterns=["fn", "fn,ln", "fn,ln,ln"],
)
punct_aug = create_pers_augmenter(
    muslim_name_dict, force_pattern_size=False, keep_name=True, patterns=["abbpunct"]
)


# randomly change 5%/15% of characters to a neighbouring key
keyboard_aug_02 = create_keyboard_augmenter(
    doc_level=1, char_level=0.02, keyboard="QWERTY_DA"
)
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

# spacing
spacing_aug_05 = create_spacing_augmenter(doc_level=1, spacing_level=0.05)
spacing_aug = create_spacing_augmenter(doc_level=1, spacing_level=1)

n = 20
# augmenter   name               n rep
augmenters = [
    # (dont_augment, "No augmentation", 1),
    # (keyboard_aug_02, "Keystroke errors 2%", n),
    # (keyboard_aug_05, "Keystroke errors 5%", n),
    # (keyboard_aug_15, "Keystroke errors 15%", n),
    # (æøå_aug, "Æøå Augmentation", 1),
    # (lower_case_aug, "Lowercase", 1),
    # (dk_aug, "Danish names", n),
    # (muslim_aug, "Muslim names", n),
    # (f_aug, "Female names", n),
    # (m_aug, "Male names", n),
    # (punct_aug, "Abbreviated first names", 1),
    # (spacing_aug_05, "Spacing Augmention 5%", n),
    # (spacing_aug, "No Spacing", 1),
]


# %% [markdown]
# # Apply functions
# Loading application functions for necessary models. No need to create one for SpaCy pipelines.

# %%
# from dev.robustness_apply_fn.apply_fn_danlp import apply_danlp_bert
# from dev.robustness_apply_fn.apply_fn_flair import apply_flair
# #from dev.robustness_apply_fn.apply_fn_polyglot import apply_polyglot
# from dev.robustness_apply_fn.apply_fn_nerda import apply_nerda

# %% [markdown]
# # Models
# A list of models to apply. To save memory the models are only loaded in one at a time.

# %%
model_dict = {
    "stanza": "da",
    "spacy_small": "da_core_news_sm",
    "spacy_medium": "da_core_news_md",
    "spacy_large": "da_core_news_lg",
    "dacy_small": "da_dacy_small_tft-0.0.0",
    "dacy_medium": "da_dacy_medium_tft-0.0.0",
    "dacy_large": "da_dacy_large_tft-0.0.0",
    # "flair" : apply_flair,
    # #"polyglot" : apply_polyglot,
    # "danlp_bert" : apply_danlp_bert,
    # "nerda_bert" : apply_nerda,
}


# %%
# to download the danlp and nerda you will have to set up a certificate:
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

# %% [markdown]
# # Performance

# %%
from pathlib import Path

Path("robustness").mkdir(parents=True, exist_ok=True)


# %%
import pandas as pd
import spacy
import spacy_stanza
import stanza


import dacy
from dacy.score import score, n_sents_score


for mdl in model_dict:
    print(f"[INFO]: Scoring model '{mdl}' using DaCy")

    # load model
    if "dacy" in mdl:
        apply_fn = dacy.load(model_dict[mdl])
    elif "spacy" in mdl:
        apply_fn = spacy.load(model_dict[mdl])
        g = spacy.prefer_gpu()
        print("GPU", g)
    elif "stanza" in mdl:
        stanza.download(model_dict[mdl])
        # Initialize the pipeline
        apply_fn = spacy_stanza.load_pipeline(model_dict[mdl])
    else:
        apply_fn = model_dict[mdl]

    i = 0
    scores = []
    for aug, nam, k in augmenters:
        print(f"\t Running augmenter: {nam}")

        scores_ = score(corpus=test, apply_fn=apply_fn, augmenters=aug, k=k)
        scores_["model"] = mdl
        scores_["augmenter"] = nam
        scores_["i"] = i
        scores.append(scores_)

        i += 1

    for n in [5, 10]:
        scores_ = n_sents_score(n_sents=n, apply_fn=apply_fn)
        scores_["model"] = mdl
        scores_["augmenter"] = f"Input size augmentation {n} sentences"
        scores_["i"] = i + 1
        scores.append(scores_)

    scores = pd.concat(scores)

    scores.to_csv(f"robustness/{mdl}_augmentation_performance_w_dep2.csv")
