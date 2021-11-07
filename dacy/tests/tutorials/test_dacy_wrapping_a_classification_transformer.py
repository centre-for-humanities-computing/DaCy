import spacy
import dacy

from dacy.subclasses import add_huggingface_model

def test_tutorial():
    import spacy
    nlp = spacy.blank("da")  # replace with your desired pipeline
    nlp = add_huggingface_model(nlp, 
                        download_name="pin/senda", # the model name on the huggingface hub
                        doc_extension="senda_trf_data", # the doc extention for transformer data e.g. including wordpieces
                        model_name="senda",  # the name of the model in the pipeline
                        category="polarity", # the category type it predicts
                        labels=["negative", "neutral", "positive"], # possible outcome labels
                        force_extension=True,
                        )

    from transformers import AutoModelForSequenceClassification

    # load and download the model
    name = "DaNLP/da-bert-tone-sentiment-polarity"
    berttone = AutoModelForSequenceClassification.from_pretrained(name, num_labels=3)

    from dacy.subclasses import ClassificationTransformer, install_classification_extensions

    labels=["positive", "neutral", "negative"]
    doc_extension = "berttone_pol_trf_data"
    category = "polarity"

    config = {
        "doc_extension_attribute": doc_extension,
        "model": {
            "@architectures": "dacy.ClassificationTransformerModel.v1",
            "name": name,
            "num_labels": len(labels),
        },
    }


    # add the relevant extentsion to the doc
    install_classification_extensions(
        category=category, labels=labels, doc_extension=doc_extension, force=True
    )

    nlp = spacy.blank("da") # dummy nlp

    clf_transformer = nlp.add_pipe(
        "classification_transformer", name="berttone", config=config
    )
    clf_transformer.model.initialize()

    texts = ["Analysen viser, at økonomien bliver forfærdelig dårlig", 
            "Jeg tror alligvel, det bliver godt"]

    docs = nlp.pipe(texts)

    for doc in docs:
        print(doc._.polarity)
        print(doc._.polarity_prop)

    # we can also examine the wordpieces used (and see the entire TransformersData)

    doc._.berttone_pol_trf_data