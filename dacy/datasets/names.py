import os
from typing import Dict
import pandas as pd


def muslim_names() -> Dict[str, str]:
    """Returns a dictionary of Muslim names.

    Returns:
        dict: A dictionary of Muslim names containing the keys "first_name" and "last_name". The list is derived from Meldgaard (2005), https://nors.ku.dk/publikationer/webpublikationer/muslimske_fornavne/.

    Example:
        >>> from dacy.datasets import muslim_names
        >>> names = muslim_names()
        >>> names["first_names"]
        >>> names["last_names"]
    """
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "lookup_tables", "names.csv"
    )
    names = pd.read_csv(path)
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


def danish_names() -> Dict[str, str]:
    """Returns a dictionary of Danish names.

    Returns:
        dict: A dictionary of Danish names containing the keys "first_name" and "last_name". The list is derived from Danmarks statistik (2021).

    Example:
        >>> from dacy.datasets import danish_names
        >>> names = danish_names()
        >>> names["first_names"]
        >>> names["last_names"]
    """
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "lookup_tables", "names.csv"
    )

    names = pd.read_csv(path)
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
