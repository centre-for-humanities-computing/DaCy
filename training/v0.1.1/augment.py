"""This is a utility script for applying the augmentation to the DaNE test set
and including the metrics in the performance scores."""

import json
import sys

sys.path.append("../..")  # import dacy

import spacy
from spacy.scorer import Scorer
from spacy.training.augment import create_lower_casing_augmenter, dont_augment
from wasabi import msg

from dacy.augmenters import (
    create_char_swap_augmenter,
    create_keyboard_augmenter,
    create_pers_augmenter,
    create_spacing_augmenter,
    create_æøå_augmenter,
)
from dacy.datasets import dane, danish_names, female_names, male_names, muslim_names
from dacy.score import n_sents_score, score

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

swap_aug2 = create_char_swap_augmenter(doc_level=1, char_level=0.02)
swap_aug5 = create_char_swap_augmenter(doc_level=1, char_level=0.05)
swap_aug15 = create_char_swap_augmenter(doc_level=1, char_level=0.15)


# Change æ=ae, ø=oe, å=aa
æøå_aug = create_æøå_augmenter(doc_level=1, char_level=1)

# lower case text
lower_case_aug = create_lower_casing_augmenter(level=1)

# spacing
spacing_aug_05 = create_spacing_augmenter(doc_level=1, spacing_level=0.05)
spacing_aug = create_spacing_augmenter(doc_level=1, spacing_level=1)


# augmenter, Name, iterations, Description
n = 20
bootstrap_text = f" As this agmentation is stochastic it is repeated {n} times to obtain a consistent estimate and the mean is provided with its standard deviation in parenthesis."
augmenters = [
    (
        dont_augment,
        "No augmentation",
        1,
        "Applies no augmentation to the DaNE test set. Using one sentence at a time as input.",
    ),
    (
        æøå_aug,
        "Æøå Augmentation",
        1,
        "This augmentation replace the æ,ø, and å with their spelling variations ae, oe and aa respectively.",
    ),
    (lower_case_aug, "Lowercase", 1, "This augmentation lowercases all text."),
    (
        spacing_aug,
        "No Spacing",
        1,
        "This augmentation removed all spacing from the text.",
    ),
    (
        punct_aug,
        "Abbreviated first names",
        1,
        "This agmentation abbreviates the first names of entities. For instance 'Kenneth Enevoldsen' would turn to 'K. Enevoldsen'.",
    ),
    (
        keyboard_aug_02,
        "Keystroke errors 2%",
        n,
        "This agmentation simulate keystroke errors by replacing 2% of keys with a neighbouring key on a Danish QWERTY keyboard."
        + bootstrap_text,
    ),
    (
        keyboard_aug_05,
        "Keystroke errors 5%",
        n,
        "This agmentation simulate keystroke errors by replacing 5% of keys with a neighbouring key on a Danish QWERTY keyboard."
        + bootstrap_text,
    ),
    (
        keyboard_aug_15,
        "Keystroke errors 15%",
        n,
        "This agmentation simulate keystroke errors by replacing 15% of keys with a neighbouring key on a Danish QWERTY keyboard."
        + bootstrap_text,
    ),
    (
        swap_aug2,
        "Character swap 2%",
        n,
        "This agmentation mistypes by swapping two adjacent characters 2% of the time."
        + bootstrap_text,
    ),
    (
        swap_aug5,
        "Character swap 5%",
        n,
        "This agmentation mistypes by swapping two adjacent characters 5% of the time."
        + bootstrap_text,
    ),
    (
        swap_aug15,
        "Character swap 15%",
        n,
        "This agmentation mistypes by swapping two adjacent characters 15% of the time."
        + bootstrap_text,
    ),
    (
        dk_aug,
        "Danish names",
        n,
        "This agmentation replace all names with Danish names derived from Danmarks Statistik (2021)."
        + bootstrap_text,
    ),
    (
        muslim_aug,
        "Muslim names",
        n,
        "This agmentation replace all names with Muslim names derived from  Meldgaard (2005)."
        + bootstrap_text,
    ),
    (
        f_aug,
        "Female names",
        n,
        "This agmentation replace all names with Danish female names derived from Danmarks Statistik (2021)."
        + bootstrap_text,
    ),
    (
        m_aug,
        "Male names",
        n,
        "This agmentation replace all names with Danish male names derived from Danmarks Statistik (2021)."
        + bootstrap_text,
    ),
    (
        spacing_aug_05,
        "Spacing Augmention 5%",
        n,
        "This agmentation replace all names with Danish male names derived from Danmarks Statistik (2021)."
        + bootstrap_text,
    ),
]


def main(model, output):
    test = dane(splits=["test"])
    spacy.prefer_gpu()
    nlp = spacy.load(model)

    scorer = Scorer(nlp)
    scores = {}
    for aug, nam, k, desc in augmenters:
        msg.info(f"Running augmenter: {nam}")

        scores_ = score(
            corpus=test,
            apply_fn=nlp,
            augmenters=aug,
            score_fn=[scorer.score],
            k=k,
            nlp=nlp,
        )
        m_d = scores_[scores_.columns].mean().to_dict()
        s_d = scores_[scores_.columns].std().to_dict()
        m_d.pop("k")
        s_d.pop("k")
        scores[nam] = {"mean": m_d, "std": s_d, "k": k}

    for n in [5, 10]:
        nam = f"Input size augmentation {n} sentences"
        msg.info(f"Running augmenter: {nam}")
        scores_ = n_sents_score(
            n_sents=n,
            apply_fn=nlp,
            score_fn=[scorer.score],
            nlp=nlp,
        )
        m_d = scores_[scores_.columns].mean().to_dict()
        s_d = scores_[scores_.columns].std().to_dict()
        m_d.pop("k")
        s_d.pop("k")
        scores[nam] = {"mean": m_d, "std": s_d, "k": 1, "desc": desc}

    with open(output, "w") as f:
        json.dump(scores, f)
    msg.good(f"Finished. Evaluation metrics saved to '{output}'")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        help="the model to evaluate",
        required=True,
    )
    parser.add_argument(
        "--output",
        type=str,
        help="the json output filename",
        required=True,
    )

    args = parser.parse_args()
    main(args.model, args.output)

    # import os
    # os.chdir("training/v0.1.0")

    # main("training/small/dane/model-best", "metrics/yolo.json")
