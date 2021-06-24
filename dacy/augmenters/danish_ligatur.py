from .character import char_replace_augmenter
from functools import partial

def make_æøå_augmenter(doc_level: float, char_level: float):
    replace_dict = {"æ": "ae", "ø": "oe", "å": "aa"}
    return partial(char_replace_augmenter, replacement = replace_dict, doc_level = doc_level, char_level = char_level)