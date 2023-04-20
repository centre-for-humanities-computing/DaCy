""" 
debugging script for training ned
"""

from pathlib import Path

from spacy.cli._util import import_code
from spacy.cli.train import train

project = Path(__file__).parent
config_path = project / "configs" / "ned.cfg"
output_path = project / "training" / "ned"

code_path = project / "scripts" / "custom_ned_functions.py"
kb_path = project / "assets" / "daned" / "knowledge_base.kb"
dev_path = project / "corpus" / "cdt" / "dev.spacy"
train_path = project / "corpus" / "cdt" / "train.spacy"

import_code(code_path)


train(
    config_path=config_path,
    output_path=output_path,
    overrides={
        "paths.train": str(train_path),
        "paths.dev": str(dev_path),
        "nlp.lang": "da",
        "paths.kb": str(kb_path),
    },
)
