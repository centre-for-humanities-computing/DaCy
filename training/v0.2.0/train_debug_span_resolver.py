"""
debugging script for training
"""

from pathlib import Path

from spacy.cli.train import train
from spacy.cli._util import import_code

project = Path(__file__).parent
# config_path = project / "configs" / "cluster_trf.cfg"
config_path = project / "configs" / "span_resolver.cfg"
output_path = project / "training" / "span_resolver"
code_path = project / "scripts" / "custom_functions.py"

train_path = project / "corpus" / "cdt" / "spans.train.spacy"
dev_path = project / "corpus" / "cdt" / "spans.dev.spacy"

import_code(code_path)


train(
    config_path=config_path,
    output_path=output_path,
    overrides={
        "paths.train": str(train_path),
        "paths.dev": str(dev_path),
        "nlp.lang": "da",
    },
)
