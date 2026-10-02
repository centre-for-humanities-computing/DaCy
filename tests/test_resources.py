import pandas as pd

from dacy import resources


def test_ods():
    entries = resources.load_ods_fullforms()
    assert isinstance(entries, pd.DataFrame)
    assert len(entries) > 200_000
    assert list(entries.columns) == ["form", "headword", "homograph", "pos", "id"]
