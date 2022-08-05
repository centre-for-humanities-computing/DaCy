from typing import Callable, List, Optional
from warnings import warn

from spacy.lang.da import Danish
from spacy.language import Language
from spacy.tokens import Doc
from spacy_transformers.data_classes import FullTransformerBatch
from spacy_wrap import ClassificationTransformer, make_classification_transformer
from thinc.api import Config, Model

DEFAULT_CONFIG_STR = """
[hatespeech_detection]
max_batch_items = 4096
doc_extension_trf_data = "is_offensive_trf_data"
doc_extension_prediction = "is_offensive"
labels = ["not offensive", "offensive"]

[hatespeech_detection.set_extra_annotations]
@annotation_setters = "spacy-transformers.null_annotation_setter.v1"

[hatespeech_detection.model]
@architectures = "spacy-wrap.ClassificationTransformerModel.v1"
name = "DaNLP/da-bert-hatespeech-detection"
tokenizer_config = {"use_fast": true}
transformer_config = {}
mixed_precision = false
grad_scaler_config = {}

[hatespeech_detection.model.get_spans]
@span_getters = "spacy-transformers.strided_spans.v1"
window = 128
stride = 96


[hatespeech_classification]
max_batch_items = 4096
doc_extension_trf_data = "hate_speech_type_trf_data"
doc_extension_prediction = "hate_speech_type"
labels = ["særlig opmærksomhed", "personangreb", "sprogbrug", "spam & indhold"]

[hatespeech_classification.set_extra_annotations]
@annotation_setters = "spacy-transformers.null_annotation_setter.v1"

[hatespeech_classification.model]
@architectures = "spacy-wrap.ClassificationTransformerModel.v1"
name = "DaNLP/da-bert-hatespeech-classification"
tokenizer_config = {"use_fast": true}
transformer_config = {}
mixed_precision = false
grad_scaler_config = {}

[hatespeech_classification.model.get_spans]
@span_getters = "spacy-transformers.strided_spans.v1"
window = 128
stride = 96
"""


DEFAULT_CONFIG = Config().from_str(DEFAULT_CONFIG_STR)


Danish.factory(
    "dacy.hatespeech_detection",
    default_config=DEFAULT_CONFIG["hatespeech_detection"],
)(make_classification_transformer)


@Danish.factory(
    "dacy.hatespeech_classification",
    default_config=DEFAULT_CONFIG["hatespeech_classification"],
)
def make_offensive_transformer(
    nlp: Language,
    name: str,
    model: Model[List[Doc], FullTransformerBatch],
    set_extra_annotations: Callable[[List[Doc], FullTransformerBatch], None],
    max_batch_items: int,
    doc_extension_trf_data: str,
    doc_extension_prediction: str,
    labels: List[str],
) -> ClassificationTransformer:

    if not Doc.has_extension("is_offensive"):
        warn(
            "The component assumes the 'is_offensive' extension is set."
            + " To set it you can run  nlp.add_pipe('dacy.hatespeech_detection')",
        )

    # TODO: Add a conditional forward such that the model isn't run is document is not emotionally laden
    clf_mdl = ClassificationTransformer(
        vocab=nlp.vocab,
        model=model,
        set_extra_annotations=set_extra_annotations,
        max_batch_items=max_batch_items,
        name=name,
        labels=labels,
        doc_extension_trf_data=doc_extension_trf_data,
        doc_extension_prediction=doc_extension_prediction,
    )

    # overwrite extension such that it return not offensive if the document is not offensive
    if Doc.has_extension("is_offensive"):

        def label_getter(doc) -> Optional[str]:
            if doc._.is_offensive == "offensive":
                prob = getattr(doc._, f"{doc_extension_prediction}_prob")
                if prob["prob"] is not None:
                    return labels[int(prob["prob"].argmax())]
            return doc._.is_offensive

        Doc.set_extension(doc_extension_prediction, getter=label_getter, force=True)
    clf_mdl.model.initialize()

    return clf_mdl
