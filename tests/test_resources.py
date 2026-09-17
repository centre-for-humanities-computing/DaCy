import pandas as pd

from dacy import datasets, resources
from dacy.datasets import names, ods


def test_names():
    for name in (
        "load_names",
        "danish_names",
        "female_names",
        "male_names",
        "muslim_names",
    ):
        assert getattr(resources, name) is getattr(datasets, name)
        assert getattr(resources, name) is getattr(names, name)
        name_list = getattr(resources, name)()
        assert isinstance(name_list["first_name"], list)
        assert len(name_list["first_name"]) > 0
        assert isinstance(name_list["last_name"], list)
        assert len(name_list["last_name"]) > 0


def test_ods():
    assert resources.load_ods_fullforms is datasets.load_ods_fullforms
    assert resources.load_ods_fullforms is ods.load_ods_fullforms
    entries = resources.load_ods_fullforms()
    assert isinstance(entries, pd.DataFrame)
    assert len(entries) > 0
    assert list(entries.columns) == ["form", "headword", "homograph", "pos", "id"]
