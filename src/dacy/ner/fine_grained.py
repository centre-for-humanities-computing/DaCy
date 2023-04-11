from typing import Callable, Literal, Optional

from spacy.lang.da import Danish
from spacy.language import Language
from spacy.tokens import Doc

import dacy


@Danish.factory(
    "dacy/ner-fine-grained",
    default_config={
        "version": None,
        "size": "medium",
        "transformer_name": "ner-transformer",
    },
)
def create_finegrained_ner_component(
    nlp: Language,
    name: str,
    size: Literal["small", "medium", "large"],
    transformer_name: str,
    version: Optional[str],
) -> Callable[[Doc], Doc]:
    """Create a fine grained NER component using the dacy models.

    Args:
        nlp: The spaCy language pipeline
        name: The name of the component
        size: The size of the model to use. Can be "small", "medium" or "large"
        transformer_name: The name of the transformer component which the NER moel will listen to
        version: The version of the model to use. If None, the latest version will be used
    """
    if version is None:
        version = dacy.get_latest_version("da_dacy_{size}_ner_fine_grained")
    nlp_ner = dacy.load(f"da_dacy_{size}_ner_fine_grained-{version}")
    nlp.add_pipe(factory_name="transformer", name=transformer_name, source=nlp_ner)
    name_, component = nlp_ner.components[-1]
    component.tok2vec.layers[0].layers[0].upstream_name = transformer_name  # type: ignore
    component.name = name  # type: ignore
    return component
