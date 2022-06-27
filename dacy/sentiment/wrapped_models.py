from typing import Callable, List, Optional
from warnings import warn

from spacy.lang.da import Danish
from spacy.language import Language
from spacy.tokens import Doc
from spacy_transformers.data_classes import FullTransformerBatch
from spacy_wrap import ClassificationTransformer, make_classification_transformer
from thinc.api import Config, Model

DEFAULT_CONFIG_STR = """
[subjectivity]
max_batch_items = 4096
doc_extension_trf_data = "subjectivity_trf_data"
doc_extension_prediction = "subjectivity"
labels = ["objective", "subjective"]

[subjectivity.set_extra_annotations]
@annotation_setters = "spacy-transformers.null_annotation_setter.v1"

[subjectivity.model]
@architectures = "spacy-wrap.ClassificationTransformerModel.v1"
name = "DaNLP/da-bert-tone-subjective-objective"
tokenizer_config = {"use_fast": true}
transformer_config = {}
mixed_precision = false
grad_scaler_config = {}

[subjectivity.model.get_spans]
@span_getters = "spacy-transformers.strided_spans.v1"
window = 128
stride = 96


[polarity]
max_batch_items = 4096
doc_extension_trf_data = "polarity_trf_data"
doc_extension_prediction = "polarity"
labels =["positive", "neutral", "negative"]

[polarity.set_extra_annotations]
@annotation_setters = "spacy-transformers.null_annotation_setter.v1"

[polarity.model]
@architectures = "spacy-wrap.ClassificationTransformerModel.v1"
name = "DaNLP/da-bert-tone-sentiment-polarity"
tokenizer_config = {"use_fast": true}
transformer_config = {}
mixed_precision = false
grad_scaler_config = {}

[polarity.model.get_spans]
@span_getters = "spacy-transformers.strided_spans.v1"
window = 128
stride = 96


[emotionally_laden]
max_batch_items = 4096
doc_extension_trf_data = "emotionally_laden_trf_data"
doc_extension_prediction = "emotionally_laden"
labels = ["emotional", "no emotion"]

[emotionally_laden.set_extra_annotations]
@annotation_setters = "spacy-transformers.null_annotation_setter.v1"

[emotionally_laden.model]
@architectures = "spacy-wrap.ClassificationTransformerModel.v1"
name = "DaNLP/da-bert-emotion-binary"
tokenizer_config = {"use_fast": true}
transformer_config = {}
mixed_precision = false
grad_scaler_config = {}

[emotionally_laden.model.get_spans]
@span_getters = "spacy-transformers.strided_spans.v1"
window = 128
stride = 96


[emotion]
max_batch_items = 4096
doc_extension_trf_data = "emotion_trf_data"
doc_extension_prediction = "emotion"
labels = ["glæde/sindsro", "tillid/accept", "forventning/interrese", "overasket/målløs", "vrede/irritation", "foragt/modvilje", "sorg/trist", "frygt/bekymret"]


[emotion.set_extra_annotations]
@annotation_setters = "spacy-transformers.null_annotation_setter.v1"

[emotion.model]
@architectures = "spacy-wrap.ClassificationTransformerModel.v1"
name = "DaNLP/da-bert-emotion-classification"
tokenizer_config = {"use_fast": true}
transformer_config = {}
mixed_precision = false
grad_scaler_config = {}

[emotion.model.get_spans]
@span_getters = "spacy-transformers.strided_spans.v1"
window = 128
stride = 96
"""


DEFAULT_CONFIG = Config().from_str(DEFAULT_CONFIG_STR)


Danish.factory(
    "dacy.subjectivity",
    default_config=DEFAULT_CONFIG["subjectivity"],
)(make_classification_transformer)


Danish.factory(
    "dacy.polarity",
    default_config=DEFAULT_CONFIG["polarity"],
)(make_classification_transformer)

Danish.factory(
    "dacy.emotionally_laden",
    default_config=DEFAULT_CONFIG["emotionally_laden"],
)(make_classification_transformer)


@Danish.factory(
    "dacy.emotion",
    default_config=DEFAULT_CONFIG["emotion"],
)
def make_emotion_transformer(
    nlp: Language,
    name: str,
    model: Model[List[Doc], FullTransformerBatch],
    set_extra_annotations: Callable[[List[Doc], FullTransformerBatch], None],
    max_batch_items: int,
    doc_extension_trf_data: str,
    doc_extension_prediction: str,
    labels: List[str],
) -> ClassificationTransformer:

    if not Doc.has_extension("dacy.emotionally_laden"):
        warn(
            "The 'emotion' component assumes the 'emotionally_laden' extension is set."
            + " To set it you can run  nlp.add_pipe('dacy.emotionally_laden')",
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

    # overwrite extension such that it return no emotion if the document does not have an emotion
    if Doc.has_extension("dacy.emotionally_laden"):

        def label_getter(doc) -> Optional[str]:
            if doc._.emotionally_laden == "emotional":
                prob = getattr(doc._, f"{doc_extension_prediction}_prob")
                if prob["prob"] is not None:
                    return labels[int(prob["prob"].argmax())]
            return doc._.emotionally_laden

        Doc.set_extension(doc_extension_prediction, getter=label_getter, force=True)
    clf_mdl.model.initialize()
    return clf_mdl
