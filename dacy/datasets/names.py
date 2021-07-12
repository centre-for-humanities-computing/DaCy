"""
Helper functions for loading name dictionaries for person augmentation.
"""

import os
from typing import Dict, List, Optional
import pandas as pd


def load_names(
    min_count: int = 0,
    ethnicity: Optional[str] = None,
    gender: Optional[str] = None,
    min_prop_gender: float = 0,
) -> Dict[str, List[str]]:
    """
    Loads the names lookup table. Danish are from Danmarks statistik (2021).
    Muslim names are from Meldgaard (2005), https://nors.ku.dk/publikationer/webpublikationer/muslimske_fornavne/.

    Args:
        min_count (int, optional): Minimum number of occurences of the name for it to be included.
            Defaults to 0.
        ethnicity (Optional[str], optional): Which ethnicity should be included. None indicate all is
            included. Options include "muslim", "danish". Defaults to None.
        gender (Optional[str], optional): Which gender should be included. None indicate all is included.
            Options include "male", "female". Defaults to None.
        min_prop_gender (float): minimum probability of a name being a given gender. The probability of a
            given name being a
            specific gender is based on the proportion of people with the given name of that gender. Only
            used when gender is set. Defaults to 0.

    Returns:
        Dict[str, List[str]]: A dictionary of Muslim names containing the keys "first_name" and "last_name".
    """
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "lookup_tables", "names.csv"
    )
    names = pd.read_csv(path)

    if min_count:
        names = names.loc[names["count"] >= min_count]

    if ethnicity is not None:
        names = names.loc[names["ethnicity"] == ethnicity]

    last_names = names.loc[names["first_name"] == False]
    if gender is not None:
        names = names.groupby(["name", "gender", "first_name"]).agg({"count": "sum"})
        # Change: groupby state_office and divide by sum
        names = names.groupby(level=0).apply(lambda x: x / float(x.sum()))
        names = names.reset_index()
        names = names.loc[
            (names["gender"] == gender) & (names["count"] >= min_prop_gender)
        ]

    first_names = names.loc[names["first_name"] == True]
    return {
        "first_name": first_names.name.tolist(),
        "last_name": last_names.name.tolist(),
    }


def muslim_names() -> Dict[str, List[str]]:
    """Returns a dictionary of Muslim names.

    Returns:
        dict: A dictionary of Muslim names containing the keys "first_name" and "last_name". The list is derived from Meldgaard (2005),
            https://nors.ku.dk/publikationer/webpublikationer/muslimske_fornavne/.

    Example:
        >>> from dacy.datasets import muslim_names
        >>> names = muslim_names()
        >>> names["first_name"]
        >>> names["last_name"]
    """
    return load_names(ethnicity="muslim")


def danish_names() -> Dict[str, List[str]]:
    """Returns a dictionary of Danish names.

    Returns:
        dict: A dictionary of Danish names containing the keys "first_name" and "last_name". The list is derived from Danmarks statistik (2021).

    Example:
        >>> from dacy.datasets import danish_names
        >>> names = danish_names()
        >>> names["first_name"]
        >>> names["last_name"]
    """
    return load_names(ethnicity="danish")


def female_names() -> Dict[str, List[str]]:
    """Returns a dictionary of Danish female names.

    Returns:
        dict: A dictionary of names containing the keys "first_name" and "last_name". The list is derived from Danmarks statistik (2021).

    Example:
        >>> from dacy.datasets import female_names
        >>> names = female_names()
        >>> names["first_name"]
        >>> names["last_name"]
    """
    return load_names(ethnicity="danish", gender="female", min_prop_gender=0.5)


def male_names() -> Dict[str, List[str]]:
    """Returns a dictionary of Danish male names.

    Returns:
        dict: A dictionary of names containing the keys "first_name" and "last_name". The list is derived from Danmarks statistik (2021).

    Example:
        >>> from dacy.datasets import male_names
        >>> names = male_names()
        >>> names["first_name"]
        >>> names["last_name"]
    """
    return load_names(ethnicity="danish", gender="male", min_prop_gender=0.5)
