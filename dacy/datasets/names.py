import os
import pandas as pd


def muslim_names() -> dict:
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


def danish_names() -> dict:
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
