"""
debugging script for training
"""

from pathlib import Path

from spacy.cli.train import train
from spacy.cli._util import import_code

project = Path(__file__).parent
# config_path = project / "configs" / "cluster_trf.cfg"
config_path = project / "configs" / "config.cfg"
output_path = project / "training" / "test"
# code_path = project / "scripts" / "custom_functions.py"

train_path = project / "corpus" / "cdt" / "train.spacy"
dev_path = project / "corpus" / "cdt" / "dev.spacy"

# import_code(code_path)

import spacy


# f"spacy train configs/config.cfg"
# + f" --output {training_path} "
# + "--paths.train corpus/cdt/train.spacy "
# + "--paths.dev corpus/cdt/dev.spacy "
# + "--nlp.lang=da "
# + f"--components.transformer.model.name={model} "
# + f"--training.logger.run_name={run_name} "
train(
    config_path=config_path,
    output_path=output_path,
    overrides={
        "paths.train": str(train_path),
        "paths.dev": str(dev_path),
        "nlp.lang": "da",
        "training.logger.run_name": "test",
        "components.transformer.model.name": "vesteinn/DanskBERT",
    },
    use_gpu=0,  # -1 for CPU
)
