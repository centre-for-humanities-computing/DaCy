import dacy
import spacy
from dacy.datasets import dane


def test_tutorial():
    # load the DaNE test set
    test = dane(splits=["test"])

    # download model
    from spacy.cli import download
    
    download("da_core_news_sm")

    # load models
    spacy_small = spacy.load("da_core_news_sm")
    dacy_small = dacy.load("small")

    from dacy.score import score

    spacy_baseline = score(test, apply_fn=spacy_small, score_fn=["ents", "pos"])
    dacy_baseline = score(test, apply_fn=dacy_small, score_fn=["ents", "pos"])

    print(spacy_baseline)
    print(dacy_baseline)

    from dacy.augmenters import create_pers_augmenter
    from dacy.datasets import female_names
    from spacy.training.augment import create_lower_casing_augmenter

    lower_aug = create_lower_casing_augmenter(level=1)
    female_name_dict = female_names()
    # Augmenter that replaces names with random Danish female names. Keep the format of the name as is (force_pattern_size=False)
    # but replace the name with one of the two defined patterns
    female_aug = create_pers_augmenter(female_name_dict, 
                                    patterns=["fn,ln","abbpunct,ln"], 
                                    force_pattern_size=False,
                                    keep_name=False)

    spacy_aug = score(test, 
                    apply_fn=spacy_small,
                    score_fn=["ents", "pos"],
                    augmenters=[lower_aug, female_aug])
    dacy_aug = score(test,
                    apply_fn=dacy_small,
                    score_fn=["ents", "pos"],
                    augmenters=[lower_aug, female_aug])

    import pandas as pd

    print(pd.concat([spacy_baseline, spacy_aug]))

    print(pd.concat([dacy_baseline, dacy_aug]))

    from dacy.augmenters import create_keyboard_augmenter

    key_05_aug = create_keyboard_augmenter(doc_level=1, char_level=0.05, keyboard="QWERTY_DA")

    spacy_key = score(test, 
                    apply_fn=spacy_small,
                    score_fn=["ents", "pos"],
                    augmenters=[key_05_aug],
                    k=5)

    print(spacy_key)

    # Excluding danlp as it is a specific version and thus hard to maintain going forward
