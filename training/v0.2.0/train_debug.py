"""
debugging script for training

debugging:
spacy train configs/cluster.cfg --output training/cluster --paths.train corpus/cdt/train.spacy --paths.dev corpus/cdt/dev.spacy --nlp.lang=da
"""

from pathlib import Path

from spacy.cli.train import train

project = Path(__file__).parent
# config_path = project / "configs" / "cluster_trf.cfg"
config_path = project / "configs" / "cluster.cfg"

output_path = project / "training" / "cluster"

train_path = project / "corpus" / "cdt" / "train.spacy"
dev_path = project / "corpus" / "cdt" / "dev.spacy"


train(
    config_path=config_path,
    output_path=output_path,
    overrides={
        "paths.train": str(train_path),
        "paths.dev": str(dev_path),
        "nlp.lang": "da",
    },
)
