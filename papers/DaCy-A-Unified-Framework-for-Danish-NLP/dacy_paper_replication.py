from pathlib import Path

import pandas as pd
import spacy
import spacy_stanza
import stanza
from spacy.training.augment import create_lower_casing_augmenter, dont_augment

import dacy
from dacy.augmenters import (
    create_keyboard_augmenter,
    create_pers_augmenter,
    create_spacing_augmenter,
    create_æøå_augmenter,
)
from dacy.datasets import dane, danish_names, female_names, male_names, muslim_names
from dacy.score import n_sents_score, score

from .apply_fns.apply_fn_danlp import apply_danlp_bert
from .apply_fns.apply_fn_flair import apply_flair
from .apply_fns.apply_fn_nerda import apply_nerda
from .apply_fns.apply_fn_polyglot import apply_polyglot

# # to download the danlp and nerda you will have to set up a certificate:
# import ssl

# ssl._create_default_https_context = ssl._create_unverified_context

# Dataset
test = dane(splits=["test"])

# Augmenters - Create a list of augmenters we wish to apply to our model.

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
    f_name_dict,
    force_pattern_size=True,
    keep_name=False,
    patterns=["fn", "fn,ln", "fn,ln,ln"],
)
m_aug = create_pers_augmenter(
    m_name_dict,
    force_pattern_size=True,
    keep_name=False,
    patterns=["fn", "fn,ln", "fn,ln,ln"],
)
punct_aug = create_pers_augmenter(
    muslim_name_dict,
    force_pattern_size=False,
    keep_name=True,
    patterns=["abbpunct"],
)


# randomly change 5%/15% of characters to a neighbouring key
keyboard_aug_02 = create_keyboard_augmenter(
    doc_level=1,
    char_level=0.02,
    keyboard="QWERTY_DA",
)
keyboard_aug_05 = create_keyboard_augmenter(
    doc_level=1,
    char_level=0.05,
    keyboard="QWERTY_DA",
)
keyboard_aug_15 = create_keyboard_augmenter(
    doc_level=1,
    char_level=0.15,
    keyboard="QWERTY_DA",
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
    (dont_augment, "No augmentation", 1),
    (keyboard_aug_02, "Keystroke errors 2%", n),
    (keyboard_aug_05, "Keystroke errors 5%", n),
    (keyboard_aug_15, "Keystroke errors 15%", n),
    (æøå_aug, "Æøå Augmentation", 1),
    (lower_case_aug, "Lowercase", 1),
    (dk_aug, "Danish names", n),
    (muslim_aug, "Muslim names", n),
    (f_aug, "Female names", n),
    (m_aug, "Male names", n),
    (punct_aug, "Abbreviated first names", 1),
    (spacing_aug_05, "Spacing Augmention 5%", n),
    (spacing_aug, "No Spacing", 1),
]

# Apply functions and models
# Loading application functions for necessary models. No need to create one for SpaCy pipelines.

model_dict = {
    "stanza": "da",
    "spacy_small": "da_core_news_sm",
    "spacy_medium": "da_core_news_md",
    "spacy_large": "da_core_news_lg",
    "dacy_small": "da_dacy_small_tft-0.0.0",
    "dacy_medium": "da_dacy_medium_tft-0.0.0",
    "dacy_large": "da_dacy_large_tft-0.0.0",
    "flair": apply_flair,
    "polyglot": apply_polyglot,
    "danlp_bert": apply_danlp_bert,
    "nerda_bert": apply_nerda,
}

# # Performance

Path("robustness").mkdir(parents=True, exist_ok=True)

for mdl in model_dict:
    print(f"[INFO]: Scoring model '{mdl}' using DaCy")

    # load model
    if "dacy" in mdl:
        apply_fn = dacy.load(model_dict[mdl])
    elif "spacy" in mdl:
        apply_fn = spacy.load(model_dict[mdl])
        spacy.prefer_gpu()
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

    scores.to_csv(f"robustness/{mdl}_augmentation_performance.csv")
