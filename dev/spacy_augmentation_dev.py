"""
This includes a series of SpaCy Augmenters
"""
from functools import partial
from math import inf
from typing import Callable, Iterator, List, Union

import spacy
from spacy.training import Example
import random
from spacy.training.augment import create_lower_casing_augmenter
from spacy.language import Language


##########
# TODO
# Make augment_entity() choose random pattern within example
##########

"""
Augmenter function
"""


@spacy.registry.augmenters("name_augmenter.v1")
def create_name_augmenter(
    ent_dict: dict,
    patterns: List[str] = ["fn,ln", "abbpunct,ln"],
    patterns_prob: List[float] = None,
    force_size: bool = False,
    keep_name: bool = True,
    prob: float = 1,
) -> Callable[[Language, Example], Iterator[Example]]:
    """Create name augmenter

    Args:
        ent_dict (dict): A dictionary of first_name and last_name to replace with
        patterns (List[float, optional): The patterns to replace with.
            Will choose one at random, optionally weighted by pattern_probs
            Options: "fn", "ln", "abb", "abbpunct". Defaults to ["fn,ln","abbpunct,ln"].
        patterns_prob (List[float]). Weights for the patterns. Defaults to None (equal weights)
        force_size (bool, optional): Whether to force entities have the same format as pattern. Defaults to False.
        keep_name (bool, optional): Whether to use the current name or sample from ent_dict. Defaults to True.
        prob (float, optional): which proportion of entities to augment. Defaults to 1.

    Returns:
        Callable[[Language, Example], Iterator[Example]]
    """
    return partial(
        name_augmenter,
        ent_dict=ent_dict,
        patterns=patterns,
        patterns_prob=patterns_prob,
        force_size=force_size,
        keep_name=keep_name,
        prob=prob,
    )


def name_augmenter(
    nlp: Language,
    example: Example,
    ent_dict: dict,
    patterns: list,
    patterns_prob: Union[bool, List[float]],
    force_size: bool,
    keep_name: bool,
    prob: float,
) -> Iterator[Example]:
    ex_dict = example.to_dict()

    # Get slices containing names
    entity_slices = get_ent_slices(ex_dict["doc_annotation"]["entities"])
    # Extract tokens corresponding to names
    name_tokens = get_slice_spans(ex_dict["token_annotation"]["ORTH"], entity_slices)

    # add possibility to sample from different patterns

    # more args
    aug_ents = augment_entity(
        entities=name_tokens,
        ent_dict=ent_dict,
        patterns=patterns,
        patterns_prob=patterns_prob,
        force_size=force_size,
        keep_name=keep_name,
        prob=prob,
    )
    # Update fields in example dictionary to match changes
    up_ex_dict = update_spacy_properties(ex_dict, aug_ents, entity_slices)
    text = make_text_from_orth(up_ex_dict)

    doc = nlp.make_doc(text)
    yield example.from_dict(doc, up_ex_dict)


"""
Name sampling functions
"""


def sample_first_name(name: str, keep_name: bool, name_dict: dict) -> str:
    if keep_name:
        return name
    else:
        return random.choice(name_dict["first_name"])


def sample_abbreviation(name: str, keep_name: bool, name_dict: dict) -> str:
    if keep_name:
        return name[0]
    else:
        return random.choice(name_dict["first_name"])[0]


def sample_abbreviation_punct(name: str, keep_name: bool, name_dict: dict) -> str:
    if keep_name:
        return name[0] + "."
    else:
        return random.choice(name_dict["first_name"])[0] + "."


def sample_last_name(name: str, keep_name: bool, name_dict: dict) -> str:
    if keep_name:
        return name
    else:
        return random.choice(name_dict["last_name"])


"""
Slicers
"""


def get_ent_slices(entities: List[str], ent_type="PER") -> List[tuple]:
    slices = []

    start = None
    end = None
    for i, ent in enumerate(entities):
        if not ent.endswith(ent_type) and end == i - 1:
            slices.append(tuple([start, end + 1]))
        if ent.endswith(ent_type):
            if ent.startswith("U"):
                slices.append(tuple([i, i + 1]))
            if ent.startswith("B"):
                start = i
            if ent.startswith("L"):
                end = i
    return slices


def get_slice_spans(l: List[str], slices: List[tuple]):
    """Get the spans corresponding to some slices"""
    return [l[slice(s[0], s[1])] for s in slices]


def update_spacy_properties(
    example_dict: dict,
    augmented_entities: List[List[str]],
    entity_slices: List[tuple],
) -> dict:

    for k, v in example_dict["token_annotation"].items():
        example_dict["token_annotation"][k] = update_slice(
            k, v, augmented_entities, entity_slices
        )
    example_dict["doc_annotation"]["entities"] = update_slice(
        "entities",
        example_dict["doc_annotation"]["entities"],
        augmented_entities,
        entity_slices,
    )
    return example_dict


def update_slice(
    type: str,
    values: List[str],
    aug_ents: List[List[str]],
    entity_slices: List[tuple],
) -> List[str]:
    if type == "ORTH":
        return handle_orth(values, aug_ents, entity_slices)
    elif type == "SPACY":
        return handle_spacy(values, aug_ents, entity_slices)
    elif type == "TAG":
        return handle_tag(values, aug_ents, entity_slices)
    elif type == "LEMMA":
        return handle_lemma(values, aug_ents, entity_slices)
    elif type == "POS":
        return handle_pos(values, aug_ents, entity_slices)
    elif type == "MORPH":
        return handle_morph(values, aug_ents, entity_slices)
    elif type == "HEAD":
        return handle_head(values, aug_ents, entity_slices)
    elif type == "DEP":
        return handle_dep(values, aug_ents, entity_slices)
    elif type == "SENT_START":
        return handle_sent_start(values, aug_ents, entity_slices)
    elif type == "entities":
        return handle_entities(values, aug_ents, entity_slices)


"""
Handlers for spacy properties
"""


def handle_orth(
    values: List[str], aug_ents: List[List[str]], entity_slices: List[tuple]
) -> List[str]:

    running_add = 0
    for i, s in enumerate(entity_slices):
        values[slice(s[0] + running_add, s[1] + running_add)] = aug_ents[i]
        running_add += len(aug_ents[i]) - (s[1] - s[0])
    return values


def handle_spacy(
    values: List[str], aug_ents: List[List[str]], entity_slices: List[tuple]
) -> List[str]:
    running_add = 0
    for i, s in enumerate(entity_slices):
        values[slice(s[0] + running_add, s[1] + running_add)] = [True] * len(
            aug_ents[i]
        )
        running_add += len(aug_ents[i]) - (s[1] - s[0])
    return values


def handle_tag(
    values: List[str], aug_ents: List[List[str]], entity_slices: List[tuple]
) -> List[str]:
    running_add = 0
    for i, s in enumerate(entity_slices):
        values[slice(s[0] + running_add, s[1] + running_add)] = ["PROPN"] * len(
            aug_ents[i]
        )
        running_add += len(aug_ents[i]) - (s[1] - s[0])
    return values


def handle_lemma(
    values: List[str], aug_ents: List[List[str]], entity_slices: List[tuple]
) -> List[str]:
    return handle_orth(values, aug_ents, entity_slices)


def handle_pos(
    values: List[str], aug_ents: List[List[str]], entity_slices: List[tuple]
) -> List[str]:
    """keep first pos tag as original, add flat to rest"""
    running_add = 0
    for i, s in enumerate(entity_slices):
        values[slice(s[0] + running_add, s[1] + running_add)] = [
            values[slice(s[0] + running_add, s[1] + running_add)][0]
        ] + ["flat"] * (len(aug_ents[i]) - 1)
        running_add += len(aug_ents[i]) - (s[1] - s[0])
    return values


def handle_morph(
    values: List[str], aug_ents: List[List[str]], entity_slices: List[tuple]
) -> List[str]:
    running_add = 0
    for i, s in enumerate(entity_slices):
        values[slice(s[0] + running_add, s[1] + running_add)] = [""] * len(aug_ents[i])
        running_add += len(aug_ents[i]) - (s[1] - s[0])
    return values


def handle_head(
    values: List[str], aug_ents: List[List[str]], entity_slices: List[tuple]
) -> List[str]:
    """keep first head, set rest to refer to index of first name"""
    running_add = 0
    for i, s in enumerate(entity_slices):
        values[slice(s[0] + running_add, s[1] + running_add)] = [
            values[slice(s[0] + running_add, s[1] + running_add)][0]
        ] + [s[0] + running_add] * (len(aug_ents[i]) - 1)
        running_add += len(aug_ents[i]) - (s[1] - s[0])
    return values


def handle_dep(
    values: List[str], aug_ents: List[List[str]], entity_slices: List[tuple]
) -> List[str]:
    return handle_pos(values, aug_ents, entity_slices)


def handle_sent_start(
    values: List[str], aug_ents: List[List[str]], entity_slices: List[tuple]
) -> List[str]:
    """keep first (if sent start), set rest to 0"""
    running_add = 0
    for i, s in enumerate(entity_slices):
        values[slice(s[0] + running_add, s[1] + running_add)] = [
            values[slice(s[0] + running_add, s[1] + running_add)][0]
        ] + [0] * (len(aug_ents[i]) - 1)
        running_add += len(aug_ents[i]) - (s[1] - s[0])
    return values


def handle_entities(
    values: List[str], aug_ents: List[List[str]], entity_slices: List[tuple]
) -> List[str]:
    running_add = 0
    for i, s in enumerate(entity_slices):
        len_ents = s[1] - s[0]
        if len_ents == 1:
            values[slice(s[0] + running_add, s[1] + running_add)] = ["U-PER"]
        else:
            values[slice(s[0] + running_add, s[1] + running_add)] = (
                ["B-PER"] + ["I-PER"] * (len_ents - 2) + ["L-PER"]
            )
        running_add += len(aug_ents[i]) - (s[1] - s[0])
    return values


"""
Augment
"""


def augment_entity(
    entities: List[List[str]],
    ent_dict: dict,
    patterns: List[str] = ["fn,ln", "abbpunct,ln"],
    patterns_prob: List[float] = None,
    force_size: bool = False,
    keep_name: bool = True,
    prob: float = 1,
) -> List[List[str]]:
    """Augment entities

    Args:
        entities (List[List[str]]): A list of lists of spans of entities/names to replace
        ent_dict (dict): A dictionary of first_name and last_name to replace with
        patterns (List[float, optional): The patterns to replace with.
            Will choose one at random, optionally weighted by pattern_probs
            Options: "fn", "ln", "abb", "abbpunct". Defaults to ["fn,ln","abbpunct,ln"].
        patterns_prob (List[float]). Weights for the patterns. Defaults to None (equal weights)
        force_size (bool, optional): Whether to force entities have the same format as pattern. Defaults to False.
        keep_name (bool, optional): Whether to use the current name or sample from ent_dict. Defaults to True.
        prob (float, optional): which proportion of entities to augment. Defaults to 1.

    Returns:
        List[List[str]]: [description]
    """
    if isinstance(patterns, str):
        patterns = [patterns]

    pattern = random.choices(patterns, weights=patterns_prob, k=1)[0]
    pattern = pattern.split(",")

    new_entity_spans = []

    for entity_span in entities:
        if force_size:
            entity_span = resize_entity_list(entity_span, pattern, ent_dict)

        new_entity = []

        for i, ent in enumerate(entity_span):
            if random.random() > prob:
                new_entity.append(ent)
            else:
                if i > len(pattern):
                    continue
                if pattern[i] == "fn":
                    new_entity.append(sample_first_name(ent, keep_name, ent_dict))
                if pattern[i] == "ln":
                    new_entity.append(sample_last_name(ent, keep_name, ent_dict))
                if pattern[i] == "abb":
                    new_entity.append(sample_abbreviation(ent, keep_name, ent_dict))
                if pattern[i] == "abbpunct":
                    new_entity.append(
                        sample_abbreviation_punct(ent, keep_name, ent_dict)
                    )
        new_entity_spans.append(new_entity)
    return new_entity_spans


def resize_entity_list(
    entity: List[str], pattern: List[str], ent_dict: dict
) -> List[str]:
    if len(entity) > len(pattern):
        return entity[: len(pattern)]
    else:
        return entity + [
            random.choice(ent_dict["last_name"])
            for _ in range(len(pattern) - len(entity))
        ]


def make_text_from_orth(example_dict: dict) -> str:
    """
    Reconstructs the text based on the changes made to ORTH
    """
    text = ""
    for orth, spacy in zip(
        example_dict["token_annotation"]["ORTH"],
        example_dict["token_annotation"]["SPACY"],
    ):
        text += orth
        if spacy:
            text += " "
    return text


"""
Testing
"""


def make_ent_dict():
    return {
        "first_name": ["Johnny", "Birte", "Tony"],
        "last_name": ["Johnnysen", "Birtesen", "Tonysen"],
    }


if _name__ == "__main__":

    ent_dict = make_ent_dict()

    nlp = spacy.load("da_core_news_sm")
    doc = nlp("Mit navn er Kenneth Henrik Enevoldsen, og Lasse, Jakob og Kenneth.")
    example = Example(doc, doc)
    ex_dict = example.to_dict()

    entity_slices = get_ent_slices(ex_dict["doc_annotation"]["entities"])
    orth = get_slice_spans(ex_dict["token_annotation"]["ORTH"], entity_slices)

    # add possibility to sample from different patterns

    aug_ents = augment_entity(orth, ent_dict, force_size=True)
    ent_lens = [len(ents) for ents in aug_ents]

    aug_ent = augment_entity(orth, ent_dict, force_size=True)

    up_ex_dict = update_spacy_properties(ex_dict, aug_ent, entity_slices)

    _orth = ex_dict["token_annotation"]["ORTH"]

    running_add = 0
    for i, s in enumerate(entity_slices):
        _orth[slice(s[0] + running_add, s[1] + running_add)] = aug_ents[i]
        running_add += len(aug_ents[i]) - (s[1] - s[0])

    _orth

    augment_entity(orth, ent_dict)
    augment_entity(orth, ent_dict, pattern="abbpunct,ln,fn,ln,ln", force_size=False)
