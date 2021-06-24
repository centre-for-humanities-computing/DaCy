"""A bunch of utilities function that can be reused across different augmenters"""
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


"""
Testing
"""


def make_test_ent_dict():
    return {
        "first_name": ["Johnny", "Birte", "Tony"],
        "last_name": ["Johnnysen", "Birtesen", "Tonysen"],
    }


def make_muslim_name_dict():
    names = pd.read_csv("../lookup_tables/names.csv")
    first_names = names.loc[
        (names["ethnicity"] == "muslim") & (names["first_name"] == True)
    ]
    last_names = names.loc[
        (names["ethnicity"] == "muslim") & (names["first_name"] == False)
    ]
    return {
        "first_name": first_names.name.tolist(),
        "last_name": last_names.name.tolist(),
    }


def make_danish_name_dict():
    names = pd.read_csv("../lookup_tables/names.csv")
    first_names = names.loc[
        (names["ethnicity"] == "danish") & (names["first_name"] == True)
    ]
    last_names = names.loc[
        (names["ethnicity"] == "danish") & (names["first_name"] == False)
    ]
    return {
        "first_name": first_names.name.tolist(),
        "last_name": last_names.name.tolist(),
    }
