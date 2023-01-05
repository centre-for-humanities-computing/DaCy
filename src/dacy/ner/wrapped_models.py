from spacy.lang.da import Danish
from spacy_wrap.pipeline_component_tok_clf import make_token_classification_transformer
from thinc.api import Config

DEFAULT_CONFIG_STR = """
[token_classification_transformer]
max_batch_items = 4096
doc_extension_trf_data = "tok_clf_trf_data"
doc_extension_prediction = "tok_clf_predictions"
predictions_to = null
labels = null
aggregation_strategy = "average"

[token_classification_transformer.set_extra_annotations]
@annotation_setters = "spacy-transformers.null_annotation_setter.v1"

[token_classification_transformer.model]
@architectures = "spacy-wrap.TokenClassificationTransformerModel.v1"
name="saattrupdan/nbailab-base-ner-scandi"
tokenizer_config = {"use_fast": true}
transformer_config = {}
mixed_precision = false
grad_scaler_config = {}


[token_classification_transformer.model.get_spans]
@span_getters = "spacy-transformers.strided_spans.v1"
window = 128
stride = 96
"""


DEFAULT_CONFIG = Config().from_str(DEFAULT_CONFIG_STR)


Danish.factory(
    "dacy/ner",
    default_config=DEFAULT_CONFIG["token_classification_transformer"],
)(make_token_classification_transformer)
