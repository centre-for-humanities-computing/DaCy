import os
import shutil

import spacy

from dacy.subclasses import ClassificationTransformer, install_classification_extensions

from danlp.download import download_model as danlp_download
from danlp.download import _unzip_process_func
from danlp.download import DEFAULT_CACHE_DIR as DANLP_DIR


def test_classification_transformer():
    """
    test if it is possible to wrap a fine-tuned transformer trained for
    sequence labelling.
    """
    texts = [
        "Analysen viser, at økonomien bliver forfærdelig dårlig",
        "Jeg tror alligvel, det bliver godt",
    ]

    # downloading model and setting a path to its location
    path_sub = danlp_download(
        "bert.polarity", DANLP_DIR, process_func=_unzip_process_func, verbose=True
    )
    path_sub = os.path.join(path_sub, "bert.pol.v0.0.1")

    labels = ["positive", "neutral", "negative"]

    doc_extension = "berttone_pol_trf_data"
    category = "polarity"

    config = {
        "doc_extension_attribute": doc_extension,
        "model": {
            "@architectures": "dacy.ClassificationTransformerModel.v1",
            "name": path_sub,
            "num_labels": len(labels),
        },
    }

    # add the relevant extentsion to the doc
    install_classification_extensions(
        category=category, labels=labels, doc_extension=doc_extension, force=True
    )

    nlp = spacy.blank("da")  # dummy nlp
    clf_transformer = nlp.add_pipe(
        "classification_transformer", name="berttone", config=config
    )

    clf_transformer.model.initialize()


    docs = list(nlp.pipe(texts))

    # test if from_disk works as intended
    clf_transformer.to_disk("clf_trf_pipe")
    test_trf = nlp.add_pipe("classification_transformer", name="from_disk")
    test_trf.from_disk("clf_trf_pipe", num_labels=3)
    shutil.rmtree("clf_trf_pipe")  # delete file after test
