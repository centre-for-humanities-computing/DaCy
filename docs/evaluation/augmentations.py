import augmenty
import spacy
from dacy.datasets import danish_names, female_names, male_names, muslim_names


def get_gender_bias_augmenters() -> dict:
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

    bias_augmenters = {
        "Danish Names": dk_aug,
        "Muslim Names": muslim_aug,
        "Male Names": male_aug,
        "Female Names": fem_aug,
    }
    return bias_augmenters


def get_robustness_augmenters(prob: float = 0.05) -> dict:
    # Spelling error augmentations
    char_swap_aug = augmenty.load("char_swap_v1", level=prob)
    tok_swap_aug = augmenty.load("token_swap_v1", level=prob)
    keystroke_aug = augmenty.load(
        "keystroke_error_v1",
        level=prob,
        keyboard="da_qwerty_v1",
    )
    start_casing_aug = augmenty.load("random_starting_case_v1", level=prob)

    sim_spelling_error_aug = augmenty.combine(
        [char_swap_aug, tok_swap_aug, keystroke_aug],
    )
    inconsistent_casing_aug = start_casing_aug

    # Synonym augmentations
    wordnet_aug = augmenty.load("wordnet_synonym_v1", level=prob, lang="da")
    nlp = spacy.load("da_core_news_lg")
    emb_aug = augmenty.load("word_embedding_v1", level=prob, nlp=nlp)

    synonym_aug = augmenty.combine([wordnet_aug, emb_aug])

    # spacing augmentations
    remove_spacing_augmenter = augmenty.load("remove_spacing_v1", level=prob)
    spacing_insertion_augmenter = augmenty.load(
        "spacing_insertion_v1",
        level=prob,
    )

    spacing_aug = augmenty.combine(
        [remove_spacing_augmenter, spacing_insertion_augmenter],
    )

    # historical spelling augmentations
    upper_noun_aug = augmenty.load("da_historical_noun_casing_v1", level=1)
    æøå_aug = augmenty.load("da_æøå_replace_v1", level=1)

    hist_spelling_aug = augmenty.combine([upper_noun_aug, æøå_aug])

    return {
        "Spelling Error": sim_spelling_error_aug,
        "Inconsistent Casing": inconsistent_casing_aug,
        "Synonym replacement": synonym_aug,
        "Inconsistent Spacing": spacing_aug,
        "Historical Spelling": hist_spelling_aug,
    }
