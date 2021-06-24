

def make_muslim_name_dict():

    names = pd.read_csv("lookup_tables/names.csv")
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

    names = pd.read_csv("lookup_tables/names.csv")
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
