import urllib
import os

from dacy.download import models_url
from dacy.load import load

def test_urls():
    for m, url in models_url.items():
        print(m)
        req = urllib.request.Request(url, method="HEAD")
        f = urllib.request.urlopen(req)
        assert f.status == 200
        print("\t Status:", f.status)
        size = int(f.headers["Content-Length"]) / 1e6
        assert size > 20
        print("\t File Size:", round(size), "mb")


def test_load():
    models = ["da_dacy_medium_tft-0.0.0"]
    for m in models:
        nlp = load(m)
        nlp("Dette er en test tekst")
