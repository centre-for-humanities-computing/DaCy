## This create the table for bias scores

import augmenty
import pandas as pd
from dacy.datasets import danish_names, female_names, male_names, muslim_names
from dacy.score import score
from spacy.training import Corpus, dont_augment


def get_augmenters() -> list:
    # augmentation
    # define pattern of augmentation
    patterns = [
        ["first_name"],
        ["first_name", "last_name"],
        ["first_name", "last_name", "last_name"],
    ]

    # define person tag for augmenters
    person_tag = "PER"

    # define all augmenters

    dk_aug = augmenty.load(
        "per_replace_v1",
        patterns=patterns,
        names=danish_names(),
        level=1,
        person_tag=person_tag,
        replace_consistency=True,
    )

    dk_aug = augmenty.load(
        "per_replace_v1",
        patterns=patterns,
        names=danish_names(),
        level=1,
        person_tag=person_tag,
        replace_consistency=True,
    )

    muslim_aug = augmenty.load(
        "per_replace_v1",
        patterns=patterns,
        names=muslim_names(),
        level=1,
        person_tag=person_tag,
        replace_consistency=True,
    )

    male_aug = augmenty.load(
        "per_replace_v1",
        patterns=patterns,
        names=male_names(),
        level=1,
        person_tag=person_tag,
        replace_consistency=True,
    )

    fem_aug = augmenty.load(
        "per_replace_v1",
        patterns=patterns,
        names=female_names(),
        level=1,
        person_tag=person_tag,
        replace_consistency=True,
    )

    augmenters = [
        ("Danish Names", dk_aug),
        ("Muslim Names", muslim_aug),
        ("Male Names", male_aug),
        ("Female Names", fem_aug),
    ]
    return augmenters


def apply_models(
    models: list,
    dataset: Corpus,
    augmenters: list,
    n_rep: int = 20,
) -> pd.DataFrame:
    rows = []
    for mdl_name, nlp in models:
        # Evaluate
        out = score(
            dataset,
            apply_fn=nlp,
            augmenters=[dont_augment],
            k=1,
            score_fn=["ents"],
        )
        out["ents_f"] = out["ents_f"] * 100
        row = {
            "Model": f"{mdl_name}",
            "Augmenter": "Baseline",
            "F1": f"{out['ents_f'].mean():.2f}",
        }
        rows.append(row)

        for aug_name, aug in augmenters:
            out = score(
                dataset,
                apply_fn=nlp,
                augmenters=[aug],
                k=n_rep,
                score_fn=["ents"],
            )
            out["ents_f"] = out["ents_f"] * 100
            row = {
                "Model": f"{mdl_name}",
                "Augmenter": aug_name,
                "F1": f"{out['ents_f'].mean():.2f} ± {out['ents_f'].std():.2f}",
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    df2 = df.pivot(index="Model", columns="Augmenter", values="F1")

    df2 = df2.reset_index()
    return df2


def create_table(
    result_df: pd.DataFrame,
    augmenters: list,
) -> pd.io.formats.style.Styler:
    nam = [("", "Model"), ("", "Baseline")] + [
        ("Augmenter", aug_name) for aug_name, _ in augmenters
    ]
    super_header = pd.MultiIndex.from_tuples(nam)
    result_df.columns = super_header

    def highlight_max(s: pd.Series) -> list:
        """Highlight the maximum in a Series with bold text."""
        is_max = s == s.max()
        return ["font-weight: bold" if v else "" for v in is_max]

    s = result_df.style.apply(highlight_max, axis=0, subset=result_df.columns[1:])

    # Add a caption
    s = s.set_caption("F1 scores for the different models and augmenters")

    # Center the header and left align the model names
    s = s.set_properties(subset=[("", "Model")], **{"text-align": "left"})
    s = s.set_properties(subset=result_df.columns[2:], **{"text-align": "right"})
    super_header_style = [{"selector": ".level0", "props": [("text-align", "center")]}]

    # Apply the CSS style to the styler
    s = s.set_table_styles(super_header_style)
    # remove the index
    s = s.hide_index()
    return s


if __name__ == "__main__":
    import dacy
    import spacy
    from dacy.datasets import dane

    sp_sm = spacy.load("da_core_news_sm")
    sp_md = spacy.load("da_core_news_md")
    dacy_sm = dacy.load("da_dacy_small_trf-0.1.0")

    models = [
        ("spaCy (da_core_news_sm)", sp_sm),
        ("spaCy (da_core_news_md)", sp_md),
        ("DaCy (da_dacy_small_trf-0.1.0)", dacy_sm),
    ]

    dataset = dane(splits="test")

    augmenters = get_augmenters()
    result_df = apply_models(models, dataset, augmenters, n_rep=20)
    s = create_table(result_df, augmenters)
    print(s.render())
