"""
# 🪐 Project Workflows: Train DaCy

Language: 🇩🇰

Datasets:
- [DaNE](https://danlp-alexandra.readthedocs.io/en/latest/docs/datasets.html?highlight=dane#dane) Danish Named Entity Corpus
- da-DDT: Danish Dependency Treebank
- DaCoref: Danish Coreference Corpus
- DaNED: Danish named entity disambiguation

These all tag a subset of the Danish Universal Dependencies corpus. Thus thus project
combined it all into one dataset.

For the DaNED we remove QIDs which does not correspond to an entity in the DaNE dataset as many of are e.g.
first names and last names. Note that the QID is still within the dataset, it is just not used for training.

This project template lets you train models for:
- 1) part-of-speech tagging
- 2) lemmatization
- 3) morphologization (morphological features)
- 4) dependency parsing
- 5) sentence segmentation
- 6) named entity recognition
- 7) coreference resolution
- 8) named entity disambiguation

The project takes care of downloading the corpus, converting it to spaCy's
format and training and evaluating the model.

## Future directions

### To do
Step my step
- [x] Train ner, dep, pos, lemma, morph with small transformer model


- [x] Add pipeline for training coref model
  - [ ] Test pipeline with Transformer (find a good small model to use)
    - Maltehb/aelaectra-danish-electra-small-cased
    - 
- [ ] Add pipeline for training NED model
- [ ] Add description of manual corrections of the dataset
- [ ] Check the status of the tokenization issue: https://github.com/explosion/spaCy/discussions/12532
- [ ] check status på "'" https://github.com/UniversalDependencies/UD_Danish-DDT/issues/4
- [ ] 
- [ ] Tilføj version af datasæt
    - særlig af DaNE (HF version)
    - DDT (github version)
    - DaCoref (ikke versioneret)
    - DaNED (ikke versioneret)
    - Dansk (HF version)
- [ ] Try NED with tok2vec instead of transformer (https://spacy.io/api/architectures#EntityLinker)
- [ ] Experiment with not using gold ents for NED

### Notes

- [ ] Corefs and NED are only available for a subset of the corpus? Would it be better to train these independently?
- [ ] DaWikiNED is not currently used, but could be used to improve the NED model. In the [paper](https://aclanthology.org/2021.crac-1.7.pdf)
it only improved the model from 0.85 to 0.86 so it might not be worth it.
- [ ] Can the entity linker model use non-entity QIDs? We have quite a few of these in the DaNED dataset.
- [ ] DANSK is currently not included. It could be added
- [ ] It would be interested to see if anything could be gained from using a multilingual approach e.g. include the english ontonotes
or Norwegian Bokmål.
- [ ] Future models to compare to:
    - Coref model: https://github.com/pandora-intelligence/crosslingual-coreference
    - Eksisterende NED model på dansk fra Alexandra
    - Coref modeller fra alexandra
    - 
- [ ] Currently frequency is estimated from the training data. It is probably just fine assuming but it is probably important to add wikipedia.
    
## Usage

It uses invoke (pyinvoke.org) for task management. Install it via:
```
pip install invoke
```

To run specific tasks you can use:
```
inv <task_name>
```

for instance, you might recreate the readme file with:

```
inv create_readme
```
"""
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from invoke import Context, Task, task
from datetime import datetime

## --- Config ------------------------------------

PROJECT = "dacy"
LANGUAGE = "da"
VERSION = "0.2.0"
PYTHON = "python"
VENV_LOCATION = ".venv"
VENV_NAME = f"{PROJECT}-{LANGUAGE}-{VERSION}"
ACTIVATE_VENV = f"source {VENV_LOCATION}/{VENV_NAME}/bin/activate"
# ACTIVATE_VENV = f"echo ''"  # when using conda
GPU_ID = 0

## --- Setup ------------------------------------


def extract_tasks() -> Dict[str, Dict[str, Any]]:
    """
    Extract tasks from the current file
    """
    tasks = [t for t in globals().values() if isinstance(t, Task)]
    tasks_mapping = {}
    for _task in tasks:
        name = _task.name
        docstring = _task.__doc__ if _task.__doc__ else ""
        docstring = docstring.split("\n\n")[0]
        docstring = " ".join([d.strip() for d in docstring.split()])
        tasks_mapping_entry: Dict[str, Any] = {"desc": docstring}
        tasks_mapping_entry["pre"] = [t.name for t in _task.pre]
        tasks_mapping_entry["post"] = [t.name for t in _task.post]
        tasks_mapping[name] = tasks_mapping_entry
    return tasks_mapping


def create_table_from_tasks() -> str:
    tasks_mapping = extract_tasks()
    pre_or_post_in_tasks = any(
        _task["pre"] or _task["post"] for _task in tasks_mapping.values()
    )
    if pre_or_post_in_tasks:
        tasks_table = "| Task | Description | Pre-process | Post-process |\n| --- | --- | --- | --- |\n"
        for name, _task in tasks_mapping.items():
            tasks_table += f"| `{name}` | {_task['desc']} | {', '.join(_task['pre'])} | {', '.join(_task['post'])} |\n"
    else:
        tasks_table = "| Task | Description |\n| --- | --- |\n"
        for name, _task in tasks_mapping.items():
            tasks_table += f"| `{name}` | {_task['desc']} |\n"
    return tasks_table


@dataclass
class Emo:
    DO = "🤖"
    GOOD = "✅"
    FAIL = "🚨"
    WARN = "🚧"
    SYNC = "🚂"
    PY = "🐍"
    CLEAN = "🧹"
    TEST = "🧪"
    COMMUNICATE = "📣"
    EXAMINE = "🔍"
    INFO = "💬"


def echo_header(msg: str):
    print(f"\n--- {msg} ---")


## --- Tasks ------------------------------------


@task
def create_readme(
    c: Context,
    overwrite: bool = False,
    filename: str = "README.md",
):
    """Creates a readme file with the project workflow from the tasks.py file

    The readme contains:
    - The docstring of the tasks.py file
    - A table of each task with its docstring
    """
    readme = Path(filename)
    if readme.exists() and (not overwrite):
        print(
            f"FAILED {Emo.FAIL} {filename} already exists to overwrite it please use the -o/--overwrite flag",
        )
        exit(0)

    template = """

{docstring}

# 📋 tasks.py

{tasks_table}
    """

    from tasks import __doc__ as docstring

    # create tables from tasks
    tasks_table = create_table_from_tasks()

    with readme.open("w") as f:
        f.write(template.format(docstring=docstring, tasks_table=tasks_table))

    print(f"{Emo.GOOD} Markdown for project workflow created at: {readme.absolute()}")


def create_venv(
    c: Context,
    name: str,
    location: str,
    python: str,
    overwrite: bool = False,
    verbose: bool = True,
) -> Path:
    """Create a virtual environment.

    Args:
        c: The invoke context
        name: The name of the virtual environment
        location: The location to create the virtual environment
        python: The python instance to use for the virtual environment
            can be the full path `~/.virtualenvs/test/bin/python` or just the
            bash shortcut `python3.8`
        overwrite: If the virtual environment already exists should it be
            overwritten
        verbose: If the command should be printed to the terminal
    """
    venv_name = Path(location) / name
    venv_name.parent.mkdir(parents=True, exist_ok=True)
    if not venv_name.exists() or overwrite:
        if verbose:
            echo_header(
                f"{Emo.DO} Creating virtual environment '{name}' using {Emo.PY}{python}",
            )
        c.run(f"{python} -m venv {venv_name}")
    else:
        if verbose:
            print(
                f"{Emo.GOOD} Virtual environment already exists, if you wish to overwrite it please use the -o/--overwrite flag",
            )
    return venv_name


@task
def check_gpu(c: Context):
    """Check if spacy gpu support is working"""
    echo_header(f"{Emo.EXAMINE} Checking for GPU")
    with c.prefix(ACTIVATE_VENV):
        out = c.run("python -c 'import spacy; spacy.require_gpu()'")
    if out.exited:
        print(f"{Emo.FAIL} GPU support is not working")
        exit(1)
    print(f"{Emo.GOOD} GPU support is working")


@task
def install(c: Context, python: Optional[str] = None, overwrite: bool = False):
    """Install the project and logs in to wandb"""
    print(sys.prefix)
    echo_header(f"{Emo.DO} Installing project")

    if python is None:
        python = PYTHON
    create_venv(
        c,
        name=VENV_NAME,
        location=VENV_LOCATION,
        python=python,
        overwrite=overwrite,
    )
    # activate the virtual environment and install the requirements
    with c.prefix(ACTIVATE_VENV):
        c.run("pip install -r requirements.txt")
    # login to wandb
    echo_header(f"{Emo.COMMUNICATE} Login to wandb")
    with c.prefix(ACTIVATE_VENV):
        c.run("wandb login")
    print(f"{Emo.GOOD} Project installed")


def download_ud_ddt(c: Context, assets_path: Path):
    c.run("git clone https://github.com/UniversalDependencies/UD_Danish-DDT")
    ddt_path = assets_path / "da_ddt"
    ddt_path.mkdir(parents=True, exist_ok=True)
    for split in ["train", "dev", "test"]:
        c.run(f"mv UD_Danish-DDT/da_ddt-ud-{split}.conllu {ddt_path}/{split}.conllu")
    c.run("rm -rf UD_Danish-DDT")


def download_da_coref(c: Context, assets_path: Path):
    coref_path = assets_path / "dacoref"
    coref_path.mkdir(parents=True, exist_ok=True)
    c.run("wget http://danlp-downloads.alexandra.dk/datasets/dacoref.zip")
    c.run(f"unzip dacoref.zip -d {coref_path}")
    c.run("rm dacoref.zip")


def download_dane(c: Context, assets_path: Path):
    # https://danlp.alexandra.dk/304bd159d5de/datasets/ddt.zip
    dane_path = assets_path / "dane"
    dane_path.mkdir(parents=True, exist_ok=True)
    c.run("wget http://danlp-downloads.alexandra.dk/datasets/ddt.zip")
    c.run(f"unzip ddt.zip -d {dane_path}")
    for split in ["train", "dev", "test"]:
        c.run(f"mv {dane_path}/ddt.{split}.conllu {dane_path}/{split}.conllu")
    c.run("rm ddt.zip")


def download_daned(c: Context, assets_path: Path):
    # http://danlp-downloads.alexandra.dk/datasets/daned.zip
    daned_path = assets_path / "daned"
    daned_path.mkdir(parents=True, exist_ok=True)
    c.run("wget http://danlp-downloads.alexandra.dk/datasets/daned.zip")
    c.run(f"unzip daned.zip -d {daned_path}")
    # all files
    files = daned_path.glob("*")

    for file in files:
        # rename the files such that they don't have the daned. prefix
        file.rename(daned_path / file.name[6:])
    c.run("rm daned.zip")


@task
def fetch_assets(c: Context, overwrite: bool = False):
    """Fetch assets for model training"""
    echo_header(f"{Emo.DO} Fetching assets")
    assets_path = Path("assets")
    if overwrite and assets_path.exists():
        shutil.rmtree(assets_path)
    assets_path.mkdir(parents=True, exist_ok=True)

    with c.prefix(ACTIVATE_VENV):
        download_ud_ddt(c, assets_path)
        download_da_coref(c, assets_path)
        download_dane(c, assets_path)
        download_daned(c, assets_path)

    print(f"{Emo.GOOD} Assets fetched")


@task
def convert(c: Context):
    """Convert the data to the correct format"""
    echo_header(f"{Emo.DO} Converting data")

    with c.prefix(ACTIVATE_VENV):
        datasets = ["da_ddt", "dane"]
        for dataset in datasets:
            output_path = Path("corpus/") / dataset
            output_path.mkdir(parents=True, exist_ok=True)
            print(output_path)
            args = "--converter conllu --merge-subtokens"
            for split in ["train", "dev", "test"]:
                c.run(
                    f"python -m spacy convert assets/{dataset}/{split}.conllu {output_path} {args}",
                )
    print(f"{Emo.GOOD} Data converted")


@task
def combine(c: Context):
    """Combine the data CDT and DDT datasets"""
    echo_header(f"{Emo.DO} Combining CDT and DDT data")
    with c.prefix(ACTIVATE_VENV):
        c.run("python scripts/combine.py")
    print(f"{Emo.GOOD} Data combined")


@task
def train(
    c: Context,
    output_path: Optional[str] = None,
    model="vesteinn/DanskBERT",
    run_name: Optional[str] = None,
    gpu_id: Optional[int] = None,
    config: Optional[str] = None,
):
    """train a model using spacy train"""
    echo_header(f"{Emo.DO} Training model")
    date = datetime.now().strftime("%Y-%m-%d")

    if output_path is None:
        # we don't want to overwrite it so we add a timestamp to the path
        training_path = Path("training") / f"{model}-{date}"
    else:
        training_path = Path(output_path)  # type: ignore

    if run_name is None:
        run_name = f"{model}-{date}"

    if gpu_id is None:
        gpu_id = GPU_ID

    if config is None:
        config = "configs/config.cfg"

    training_path.mkdir(parents=True, exist_ok=True)
    cmd = (
        f"spacy train {config}"
        + f" --output {training_path} "
        + "--paths.train corpus/cdt/train.spacy "
        + "--paths.dev corpus/cdt/dev.spacy "
        + "--nlp.lang=da "
        + f"--components.transformer.model.name={model} "
        + f"--training.logger.run_name={run_name} "
        + f"--gpu-id={gpu_id} "
    )
    with c.prefix(ACTIVATE_VENV):
        c.run(cmd)
    print(f"{Emo.GOOD} Model trained")


@task
def evaluate(
    c: Context,
    model_path: str = "training/test/model-best",
    dataset: str = "dane",
):
    """Evaluate a model using spacy evaluate"""
    echo_header(f"{Emo.DO} Evaluating model")

    _model_path = Path(model_path)
    metrics_path = Path("metrics") / dataset
    test_set = Path("corpus") / dataset / "test.spacy"

    model_name = _model_path.parent.name
    model_type = _model_path.name
    metrics_json = metrics_path / f"{model_name}_{model_type}.json"

    metrics_path.mkdir(parents=True, exist_ok=True)

    with c.prefix(ACTIVATE_VENV):
        c.run(f"spacy evaluate {_model_path} {test_set} --output {metrics_json}")
    print(f"{Emo.GOOD} Model evaluated")


@task
def train_coref_cluster(c: Context):
    """
    Train the clustering component
    """
    # "python -m spacy train config/cluster.cfg -g ${vars.gpu_id} --paths.train corpus/train.spacy --paths.dev corpus/dev.spacy -o training/cluster --training.max_epochs ${vars.max_epochs}"
    echo_header(f"{Emo.DO} Training model")

    training_path = Path("training/cluster")
    training_path.mkdir(parents=True, exist_ok=True)
    with c.prefix(ACTIVATE_VENV):
        c.run(
            f"spacy train configs/cluster_trf.cfg --output {training_path} --paths.train corpus/cdt/train.spacy --paths.dev corpus/cdt/dev.spacy --nlp.lang=da --gpu-id {GPU_ID}",
        )
    print(f"{Emo.GOOD} Model trained")


@task
def prep_span_data(
    c: Context, heads="silver", model_path="training/cluster/model-best"
) -> None:
    """Prepare data for the span resolver component.

    Args:
        heads: Whether to use gold heads or silver heads predicted by the clustering component

    """
    # python scripts/prep_span_data.py --heads ${vars.heads} --model-path training/cluster/model-best/ --gpu ${vars.gpu_id} --input-path corpus/train.spacy --output-path corpus/spans.train.spacy --head-prefix coref_head_clusters --span-prefix coref_clusters
    #       - python scripts/prep_span_data.py --heads ${vars.heads} --model-path training/cluster/model-best/ --gpu ${vars.gpu_id} --input-path corpus/dev.spacy --output-path corpus/spans.dev.spacy --head-prefix coref_head_clusters --span-prefix coref_clusters

    echo_header(f"{Emo.DO} Preparing span data")

    training_path = Path("training/cluster")
    training_path.mkdir(parents=True, exist_ok=True)

    cmd = (
        "python scripts/prep_span_data.py"
        + f" --heads {heads} "
        + f"--model-path {model_path} "
        + "--input-path corpus/cdt/{split}.spacy "
        + "--output-path corpus/cdt/spans.{split}.spacy "
        + "--head-prefix coref_head_clusters "
        + "--span-prefix coref_clusters"
    )
    with c.prefix(ACTIVATE_VENV):
        for split in ["train", "dev"]:
            c.run(cmd.format(split=split))
    print(f"{Emo.GOOD} Span data prepared")


@task
def train_span_resolver(
    c: Context,
    tok2vec_source: Optional[str] = "training/cluster/model-best",
    transformer_source: Optional[str] = None,
    max_epochs: int = 20,
):
    """
    Train the span resolver component.
    """
    # spacy train configs/span.cfg -c scripts/custom_functions.py -g ${vars.gpu_id} --paths.train corpus/spans.train.spacy --paths.dev corpus/spans.dev.spacy --training.max_epochs ${vars.max_epochs} --paths.transformer_source training/cluster/model-best -o training/span_resolver
    echo_header(f"{Emo.DO} Training span resolver")

    if tok2vec_source and transformer_source:
        raise ValueError(
            "Only one of `tok2vec_source` and `transformer_source` can be set"
        )
    if tok2vec_source is None and transformer_source is None:
        raise ValueError("One of `tok2vec_source` and `transformer_source` must be set")

    config = (
        "configs/span_resolver.cfg"
        if tok2vec_source
        else "configs/span_resolver_trf.cfg"
    )

    cmd = (
        f"spacy train {config}"
        + " -c scripts/custom_functions.py"
        + " --output training/span_resolver"
        + " --paths.train corpus/cdt/spans.train.spacy"
        + " --paths.dev corpus/cdt/spans.dev.spacy"
        + f" --training.max_epochs {max_epochs}"
    )
    if tok2vec_source:
        cmd += f" --paths.tok2vec_source {tok2vec_source}"
    else:
        cmd += f" --paths.transformer_source {transformer_source}"

    with c.prefix(ACTIVATE_VENV):
        c.run(cmd)
    print(f"{Emo.GOOD} Span resolver trained")


@task
def assemple_coref(c: Context):
    """Assemble the coreference model."""
    # spacy assemble ${vars.config_dir}/coref.cfg training/coref
    echo_header(f"{Emo.DO} Assembling coreference model")

    with c.prefix(ACTIVATE_VENV):
        c.run("spacy assemble configs/coref.cfg training/coref")
    print(f"{Emo.GOOD} Coreference model assembled")


@task
def workflow_prepare_to_train(c: Context):
    """Runs: `install` &rarr; `fetch-assets` &rarr; `convert` &rarr; `combine` &rarr; `create-knowledge-base`"""
    install(c)
    fetch_assets(c)
    convert(c)
    combine(c)
    create_knowledge_base(c)

@task
def workflow_train_coref_model(c: Context):
    """Runs: `train_coref_cluster` &rarr; `prep_span_data` &rarr; `train_span_resolver` &rarr; `assemple_coref`"""
    train_coref_cluster(c)
    prep_span_data(c)
    train_span_resolver(c)
    assemple_coref(c)


@task
def create_knowledge_base(c: Context, model="vesteinn/DanskBERT") -> None:
    """Create the Knowledge Base in spaCy and write it to file."""
    # script:
    #   - "python ./scripts/create_kb.py ./assets/${vars.entities} ${vars.vectors_model} ./temp/${vars.kb} ./temp/${vars.nlp}/"
    echo_header(f"{Emo.DO} Creating Knowledge Base")

    with c.prefix(ACTIVATE_VENV):
        c.run(f"python ./scripts/create_kb.py {model}")

    print(f"{Emo.GOOD} Knowledge Base created")


@task
def train_ned(c: Context):
    """Train the named entity disambiguation component."""
    echo_header(f"{Emo.DO} Training NED model")
    # python -m spacy train configs/${vars.config}
    # --output training
    # --paths.train corpus/${vars.train}.spacy
    # --paths.dev corpus/${vars.dev}.spacy
    # --paths.kb temp/${vars.kb}
    # --paths.base_nlp temp/${vars.nlp}
    # -c scripts/custom_functions.py

    cmd = (
        "python -m spacy train configs/ned.cfg"
        + " --output training/ned"
        + " --paths.train corpus/cdt/train.spacy"
        + " --paths.dev corpus/cdt/dev.spacy"
        + " --paths.kb assets/daned/knowledge_base.kb"
        # + " --paths.base_nlp my_nlp"
        + " -c scripts/custom_ned_functions.py"
        + " --gpu-id 0"
    )

    with c.prefix(ACTIVATE_VENV):
        c.run(cmd)
    print(f"{Emo.GOOD} NED model trained")


@task
def evaluate_ned(c: Context):
    """Evaluate the named entity disambiguation component."""
    # "python ./scripts/evaluate.py ./training/model-best/ corpus/${vars.dev}.spacy"
    echo_header(f"{Emo.DO} Evaluating NED model")

    with c.prefix(ACTIVATE_VENV):
        c.run(
            "python ./scripts/evaluate_ned.py ./training/ned/model-best/ corpus/cdt/dev.spacy"
        )

    print(f"{Emo.GOOD} NED model evaluated")


@task
def workflow_train_ned(c: Context):
    """Runs: `create_knowledge_base` &rarr; `train_ned` &rarr; `evaluate_ned`"""
    create_knowledge_base(c)
    train_ned(c)
    evaluate_ned(c)
