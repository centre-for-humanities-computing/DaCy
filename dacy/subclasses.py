from typing import List, Callable, Iterable, Iterator, Optional, Dict, Union
from pathlib import Path

from spacy_transformers import Transformer
from spacy.pipeline.pipe import deserialize_config
from spacy import util

from thinc.api import get_current_ops, CupyOps

from transformers import AutoTokenizer, AutoModelForSequenceClassification

@Language.factory(
    "transformer",
    assigns=[f"doc._.{DOC_EXT_ATTR}"],
    default_config=DEFAULT_CONFIG["transformer"],
)
def make_transformer(
    nlp: Language,
    name: str,
    model: Model[List[Doc], FullTransformerBatch],
    set_extra_annotations: Callable[[List[Doc], FullTransformerBatch], None],
    max_batch_items: int,
):
    """Construct a Transformer component, which lets you plug a model from the
    Huggingface transformers library into spaCy so you can use it in your
    pipeline. One or more subsequent spaCy components can use the transformer
    outputs as features in its model, with gradients backpropagated to the single
    shared weights.
    model (Model[List[Doc], FullTransformerBatch]): A thinc Model object wrapping
        the transformer. Usually you will want to use the TransformerModel
        layer for this.
    set_extra_annotations (Callable[[List[Doc], FullTransformerBatch], None]): A
        callback to set additional information onto the batch of `Doc` objects.
        The doc._.trf_data attribute is set prior to calling the callback.
        By default, no additional annotations are set.
    """
    return Transformer(
        nlp.vocab,
        model,
        set_extra_annotations,
        max_batch_items=max_batch_items,
        name=name,
    )


class ClassificationTransformer(Transformer):
    """
    """
    def from_disk(
        self, path: Union[str, Path], *, exclude: Iterable[str] = tuple()
    ) -> "Transformer":
        """Load the pipe from disk.
        path (str / Path): Path to a directory.
        exclude (Iterable[str]): String names of serialization fields to exclude.
        RETURNS (Transformer): The loaded object.
        DOCS: https://spacy.io/api/transformer#from_disk
        """

        def load_model(p):
            p = Path(p).absolute()
            tokenizer, transformer = huggingface_classification_from_pretrained(
                p, self.model.attrs["tokenizer_config"]
            )
            self.model.attrs["tokenizer"] = tokenizer
            self.model.attrs["set_transformer"](self.model, transformer)

        deserialize = {
            "vocab": self.vocab.from_disk,
            "cfg": lambda p: self.cfg.update(deserialize_config(p)),
            "model": load_model,
        }
        util.from_disk(path, deserialize, exclude)
        return self

def huggingface_classification_from_pretrained(source: Union[Path, str], config: Dict):
    """Create a Huggingface transformer model from pretrained weights. Will
    download the model if it is not already downloaded.
    source (Union[str, Path]): The name of the model or a path to it, such as
        'bert-base-cased'.
    config (dict): Settings to pass to the tokenizer.
    """
    if hasattr(source, "absolute"):
        str_path = str(source.absolute())
    else:
        str_path = source
    tokenizer = AutoTokenizer.from_pretrained(str_path, **config)
    transformer = AutoModelForSequenceClassification.from_pretrained(str_path)
    ops = get_current_ops()
    if isinstance(ops, CupyOps):
        transformer.cuda()
    return tokenizer, transformer