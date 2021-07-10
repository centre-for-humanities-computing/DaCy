"""Utility functions used for augmentation."""
import pandas as pd


def make_text_from_orth(example_dict: dict) -> str:
    """
    Reconstructs the text based on ORTH and SPACY from an Example turned to dict
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
