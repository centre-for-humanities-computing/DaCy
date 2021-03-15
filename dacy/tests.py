import urllib
import os

from download import models_url
from load import load

def test_urls():
    for m, url in models_url.items():
        print(m)
        req = urllib.request.Request(url, method="HEAD")
        f = urllib.request.urlopen(req)
        print("\t Status:", f.status)
        size = int(f.headers["Content-Length"]) / 1e6
        print("\t File Size:", round(size), "mb")


def test_load():
    for m, url in models_url.items():
        nlp = load(m)
        nlp("Dette er en test tekst")
