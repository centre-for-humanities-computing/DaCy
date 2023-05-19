import augmenty
import spacy

prob = 0.02

# Spelling error augmentations
char_swap_aug = augmenty.load("char_swap_v1", level=prob)
tok_swap_aug = augmenty.load("token_swap_v1", level=prob)
keystroke_aug = augmenty.load("keystroke_error_v1", level=prob, keyboard="da_qwerty_v1")
start_casing_aug = augmenty.load("random_starting_case_v1", level=prob)
# common spelling error
# https://huggingface.co/datasets/KennethEnevoldsen/ddo-misspellings/tree/main
# replace = {"act": {"VERB": ["perform", "move"], "NOUN": ["action", "deed"]}}
# create_token_dict_replace_augmenter(replace=replace, level=.10)

sim_spelling_error_aug = augmenty.combine(
    [char_swap_aug, tok_swap_aug, keystroke_aug, start_casing_aug],
)


# Synonym augmentations
wordnet_aug = augmenty.load("wordnet_synonym_v1", level=prob, lang="da")
nlp = spacy.load("da_core_news_lg")
emb_aug = augmenty.load("word_embedding_v1", level=prob, nlp=nlp)

synonym_aug = augmenty.combine([wordnet_aug, emb_aug])


# spacing augmentations
remove_spacing_augmenter = augmenty.load("remove_spacing_v1", level=0.02)
spacing_insertion_augmenter = augmenty.load(
    "spacing_insertion_augmenter_v1",
    level=0.02,
)

spacing_aug = augmenty.combine([remove_spacing_augmenter, spacing_insertion_augmenter])

# historical spelling augmentations
upper_noun_aug = augmenty.load("da_historical_noun_casing_augmenter_v1", level=0.02)
æøå_aug = augmenty.load("da_æøå_replace_v1", level=0.1)

hist_spelling_aug = augmenty.combine([upper_noun_aug, æøå_aug])


# Abbreviations
def abbreviate_dot(token):
    return token.text[0] + "."


def abbreviate(token):
    return token.text[0]


ent_abbr_dot = augmenty.load(
    "ents_format_v1",
    reordering=[-1, None],
    formatter=[None, abbreviate_dot],
    level=0.1,
    ent_types=["PER"],
)
ent_abbr = augmenty.load(
    "ents_format_v1",
    reordering=[-1, None],
    formatter=[None, abbreviate],
    level=0.1,
    ent_types=["PER"],
)
last_name_only = augmenty.load(
    "ents_format_v1",
    reordering=[-1],
    formatter=[None, abbreviate],
    level=0.1,
    ent_types=["PER"],
)
first_name_only = augmenty.load(
    "ents_format_v1",
    reordering=[1],
    formatter=[None, abbreviate],
    level=0.1,
    ent_types=["PER"],
)

abbreviations_aug = augmenty.combine(
    [ent_abbr_dot, ent_abbr, last_name_only, first_name_only],
)


# sentence subset augmentations
sent_subset = augmenty.load("sent_subset_v1", level=1)
