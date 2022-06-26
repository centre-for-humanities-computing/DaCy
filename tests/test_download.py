import urllib

import dacy
from dacy.download import download_model, models_url
from dacy.load import load


def test_urls():
    for m in dacy.models():
        url = models_url[m]
        req = urllib.request.Request(url, method="HEAD")
        f = urllib.request.urlopen(req)
        assert f.status == 200

    for m in [
        "da_dacy_small_tft-0.0.0",
        "da_dacy_medium_tft-0.0.0",
        "da_dacy_large_tft-0.0.0",
    ]:
        print("\t Status:", f.status)
        size = int(f.headers["Content-Length"]) / 1e6
        assert size > 20
        print("\t File Size:", round(size), "mb")


def test_load():
    models = ["da_dacy_medium_trf-0.1.0", "da_dacy_medium_tft-0.0.0"]
    for m in models:
        nlp = load(m)
        nlp("Dette er en test tekst")


def test_models():
    print(dacy.models())


def test_where_is_my_dacy():
    print(dacy.where_is_my_dacy())


def test_download_model_error():
    download_model(model="da_dacy_medium_tft-0.0.0", force=True)
    # this just tests if it fails when needed.
    try:
        download_model(model="not a dacy model")
    except ValueError:
        pass
