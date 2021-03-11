
import os
import requests

import urllib.request

from tqdm import tqdm

url = 'https://sciencedata.dk/themes/deic_theme_oc7/apps/files_sharing/public.php?service=files&t=0e5d0b97fbead07d1f2ba7c3cbea03eb&path=%2FDaCy&files=da_dacy_medium_tft-0.0.0.tar.gz&download&g='


save_path = "/Users/au561649/Desktop/"
target_path = 'da_dacy_medium_tft-0.0.0.tar.gz'

target_path = os.path.join(save_path, target_path)

import spacy
model = spacy.load("en_core_web_sm")
model._path

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


download_url(url, target_path)