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
Step by step:
- [x] Train ner, dep, pos, lemma, morph with small transformer model
- [x] Performed manual correction of the dataset:
    - [x] Fixed issues with "'", see github issue on UD_Danish-DDT (https://github.com/UniversalDependencies/UD_Danish-DDT/issues/4)
    - [x] Combined datasets (DDT, DaNE, CDT, DaNED, DaCoref). Approach:
        - [x] Load DDT
        - [x] DaNE match 1-1 with DDT, so simply add the DaNE annotations to the DDT annotations
        - [x] CDT is a subset of the DDT, but with a different and notably including document annotations. So we
            - [x] combine each of the documents in DDT according to the annotations in CDT, the sentences without document annotations are ignored.
            - [x] overwrite the CDT split with the split from DDT (i.e. the CDT split is ignored)
        - [x] CDT contains coreference annotations (DaCoref). All of these are directly added.
        - [x] CDT also contains NED annotations (DaNED) using QID's for every possible entities (even obscure ones like, i.e.
          [mette](https://www.wikidata.org/wiki/Q1158302), refering to the name). To remove these we filter these out by
            only keeping the QID's which match an entire entity (i.e. no entity can have multiple QIDs).
          - [ ] I should probably do this in a smarter way.
        - [x] Write two datasets, one which is the extended DDT, another one which is the CDT only.
- [x] Add pipeline for training NED model
- [x] Add pipeline for training coref model
  - [x] Add pipeline for training clustering component
  - [x] Add span resolver
  - [x] Assemble it into a pipeline
- [x] Check the status of the tokenization issue: https://github.com/explosion/spaCy/discussions/12532
- [x] Try NED with tok2vec instead of transformer
- [ ] Train full pipeline
    - [ ] Train two sets of models for DaCy (one for the testing and one trained on the full dataset)
- [ ] Check the whether you can use the parser for annotation in coreference
- [ ] Save current version of the datasets
- [ ] Add scores for lemmatization
- [ ] Create a grid search for:
    - nlp
        - batch_size
    - components
        - coref
            - dropout
            - depth
            - hidden_size
            - antecedent_limit
            - antecedent_batch_size
            - grad_factor
            - tok2vec pooling
        - entity_linker
            - incl_context
            - incl_prior
            - (entity_vector_length)
            - n_sents
            - use_gold_ents
            - model.tok2vec
                - transformer vs tok2vec
                - transformer
                    - pooling
                    - grad_factor
        - ner.model
            - architectures (transition based parser vs spancat)
            - hidden_width
            - maxout_pieces
            - use_upper
            - tok2vec.pooling
        - trainable_lemmatizer
            - backoff
            - min_tree_freq
            - top_k
        - transformer
            - model.get_spans
                - window
                - stride
            - using freezed transformer
    - training
        - accumulate_gradient
        - dropout
        - patience
        - max_epochs
        - seed
        - optimizer
            - L2_is_weight_decay
            - L2
            - grad_clip
            - use_averages
            - beta1
            - beta2
        -
    - using seperate NED components

- [ ] Future models to compare to:
    - Coref model: https://github.com/pandora-intelligence/crosslingual-coreference
    - Eksisterende NED model på dansk fra Alexandra
    - Coref modeller fra alexandra
- [ ] Try to test generalization of the model by first training with the transformer frozen and then with the whole thing unfrozen: https://www.linkedin.com/feed/update/urn:li:activity:7063880655142604803/
1) https://magazine.sebastianraschka.com/p/finetuning-large-language-models
2) https://arxiv.org/abs/2202.10054

### Notes
- Corefs and NED are only available for a subset of the corpus? Would it be better to train these independently? It might be better to train them
independentently
- DaWikiNED is not currently used, but could be used to improve the NED model. In the [paper](https://aclanthology.org/2021.crac-1.7.pdf)
it only improved the model from 0.85 to 0.86 so it might not be worth it.
- The current entity linker model onyl uses entity QIDs, but the DaNED dataset contains a lot of non-entity QIDs. It might be worth it to
include these as well during a seperate training step.
- DANSK is currently not included.
- It would be interested to see if anything could be gained from using a multilingual approach e.g. include the english ontonotes
or Norwegian Bokmål.
- Currently frequency is estimated from the training data. It is probably better to also add wikipedia to this.
- The current NED annotation contains quite a new odd mentions e.g. where a person (i.e. "kenneth") has the QID which refers to the name. That is
probably wrong unless you of-course are talking about the name. It might be worth it to filter these out.

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
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Literal, Optional

from invoke import Context, Task, task

## --- Config ------------------------------------

PROJECT = "dacy"
LANGUAGE = "da"
VERSION = "0.2.0"
# path to python, we recommend using a virtual environment
PYTHON = "/home/kenneth/miniconda3/envs/dacy/bin/python"  # server
# PYTHON = "/Users/au561649/Github/DaCy/training/v0.2.0/.venv/dacy-da-0.2.0/bin/python"  # local
VENV_LOCATION = ".venv"
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


@task
def create_venv(
    c: Context,
    location: str,
    python: str,
    overwrite: bool = False,
    verbose: bool = True,
) -> Path:
    """Create a virtual environment. Optional.

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
    if python is None:
        python = PYTHON
    venv_name = Path(location)
    venv_name.parent.mkdir(parents=True, exist_ok=True)
    if not venv_name.exists() or overwrite:
        if verbose:
            echo_header(
                f"{Emo.DO} Creating virtual environment '{venv_name}' using {Emo.PY}{python}",
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
    out = c.run(f"{PYTHON} -c 'import spacy; spacy.require_gpu()'")
    if out.exited:
        print(f"{Emo.FAIL} GPU support is not working")
        exit(1)
    print(f"{Emo.GOOD} GPU support is working")


@task
def install(c: Context):
    """Install the project and logs in to wandb"""
    echo_header(f"{Emo.DO} Installing project")

    # activate the virtual environment and install the requirements
    c.run(f"{PYTHON} -m pip install -r requirements.txt")
    # login to wandb
    echo_header(f"{Emo.COMMUNICATE} Login to wandb")
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
    """Fetch assets for model training."""
    echo_header(f"{Emo.DO} Fetching assets")
    assets_path = Path("assets")
    if overwrite and assets_path.exists():
        shutil.rmtree(assets_path)
    assets_path.mkdir(parents=True, exist_ok=True)

    download_ud_ddt(c, assets_path)
    download_da_coref(c, assets_path)
    download_dane(c, assets_path)
    download_daned(c, assets_path)

    print(f"{Emo.GOOD} Assets fetched")


@task
def convert(c: Context):
    """Convert the data to the correct format"""
    echo_header(f"{Emo.DO} Converting data")

    datasets = ["da_ddt", "dane"]
    for dataset in datasets:
        output_path = Path("corpus/") / dataset
        output_path.mkdir(parents=True, exist_ok=True)
        print(output_path)
        args = "--converter conllu --merge-subtokens"
        for split in ["train", "dev", "test"]:
            c.run(
                f"{PYTHON} -m spacy convert assets/{dataset}/{split}.conllu {output_path} {args}",
            )
    print(f"{Emo.GOOD} Data converted")


@task
def combine(c: Context):
    """Combine the data CDT and DDT datasets"""
    echo_header(f"{Emo.DO} Combining CDT and DDT data")
    c.run(f"{PYTHON} scripts/create_ddt_compatible_splits_for_cdt.py")
    c.run(f"{PYTHON} scripts/combine.py")
    print(f"{Emo.GOOD} Data combined")


@task
def create_knowledge_base(c: Context, model="vesteinn/DanskBERT") -> None:
    """Create the Knowledge Base in spaCy and write it to file."""
    echo_header(f"{Emo.DO} Creating Knowledge Base")

    c.run(
        f"{PYTHON} ./scripts/create_kb.py {model} --save-path-kb assets/knowledge_bases/{model}.kb",
    )

    print(f"{Emo.GOOD} Knowledge Base created")


## --- Training ------------------------------------


@task
def train(
    c: Context,
    file_path: Optional[str] = None,
    model="vesteinn/DanskBERT",
    run_name: Optional[str] = None,
    gpu_id: Optional[int] = None,
    config: Optional[str] = None,
    overwrite: bool = False,
    dataset: Literal["cdt", "ddt", "dane"] = "cdt",
):
    """
    train a model using spacy train

    Args:
        embedding_size: The size of the transformer embedding. If None the default size is used.
    """
    echo_header(f"{Emo.DO} Training model")
    date = datetime.now().strftime("%Y-%m-%d")

    if run_name is None:
        run_name = f"{model}-{date}"
    if file_path is None:
        training_path = Path("training") / run_name
    else:
        training_path = Path(file_path)  # type: ignore

    if gpu_id is None:
        gpu_id = GPU_ID

    if config is None:
        config = "configs/config.cfg"

    if training_path.exists() and (not overwrite):
        print(
            f"{Emo.FAIL} Training path already exists to overwrite it please use the -o/--overwrite flag",
        )
        exit(1)
    training_path.mkdir(parents=True, exist_ok=True)

    cmd = (
        f"spacy train {config}"
        + f" --output {training_path} "
        + f"--paths.train corpus/{dataset}/train.spacy "
        + f"--paths.dev corpus/{dataset}/dev.spacy "
        + "--nlp.lang=da "
        + f"--gpu-id={gpu_id} "
        + f"--components.transformer.model.name={model} "
        + f"--training.logger.run_name={run_name} "
        + f"--paths.kb assets/knowledge_bases/{model}.kb "
    )

    print(f"{Emo.INFO} Running command:")
    print(cmd)
    c.run(cmd)
    print(f"{Emo.GOOD} Model trained")


@task
def prep_span_data(c: Context, run_name: str, heads="silver", bool=False) -> None:
    """Prepare data for the span resolver component.

    Args:
        heads: Whether to use gold heads or silver heads predicted by the clustering component

    """
    # python scripts/prep_span_data.py --heads ${vars.heads} --model-path training/cluster/model-best/ --gpu ${vars.gpu_id} --input-path corpus/train.spacy --output-path corpus/spans.train.spacy --head-prefix coref_head_clusters --span-prefix coref_clusters
    #       - python scripts/prep_span_data.py --heads ${vars.heads} --model-path training/cluster/model-best/ --gpu ${vars.gpu_id} --input-path corpus/dev.spacy --output-path corpus/spans.dev.spacy --head-prefix coref_head_clusters --span-prefix coref_clusters

    echo_header(f"{Emo.DO} Preparing span data")

    training_path = Path("training")
    training_path.mkdir(parents=True, exist_ok=True)
    model_path = training_path / run_name

    model_path = model_path / "model-best" if model_best else model_path / "model-last"

    cmd = (
        f"{PYTHON} scripts/prep_span_data.py"
        + f" --heads {heads} "
        + f"--model-path {model_path} "
        + "--input-path corpus/cdt/{split}.spacy "
        + "--output-path corpus/cdt/spans.{run_name}.{split}.spacy "
        + "--head-prefix coref_head_clusters "
        + "--span-prefix coref_clusters "
    )

    for split in ["train", "dev"]:
        c.run(cmd.format(split=split, run_name=run_name))
    print(f"{Emo.GOOD} Span data prepared")


@task
def train_span_resolver(
    c: Context,
    run_name: str,
    model_best: bool = False,
    gpu_id: Optional[int] = None,
):
    """
    Train the span resolver component.
    """
    echo_header(f"{Emo.DO} Training span resolver")

    training_path = Path("training")
    training_path.mkdir(parents=True, exist_ok=True)
    model_path = training_path / run_name

    model_path = model_path / "model-best" if model_best else model_path / "model-last"

    if gpu_id is None:
        gpu_id = GPU_ID

    config = "configs/span_resolver.cfg"

    cmd = (
        f"spacy train {config}"
        + " -c scripts/custom_functions.py"
        + f" --output training/{run_name}.span_resolver"
        + f" --paths.train corpus/cdt/spans.{run_name}.train.spacy"
        + f" --paths.dev corpus/cdt/spans.{run_name}.dev.spacy"
        + " --nlp.lang=da"
        + f" --training.logger.run_name={run_name}.span_resolver"
        + f" --paths.transformer_source {model_path}"
        + f" --gpu-id {gpu_id}"
    )
    c.run(cmd)
    print(f"{Emo.GOOD} Span resolver trained")


## --- Evaluate ------------------------------------


@task
def evaluate(
    c: Context,
    run_name: str,
    split: Literal["train", "dev", "test"] = "test",
    gpu_id: Optional[int] = None,
    overwrite: bool = False,
):
    """Evaluate a model using spacy evaluate"""
    echo_header(f"{Emo.DO} Evaluating model")

    training_path = Path("training")
    training_path.mkdir(parents=True, exist_ok=True)
    model_path = training_path / run_name

    if gpu_id is None:
        gpu_id = GPU_ID

    cmd = (
        f"{PYTHON} scripts/evaluate.py {model_path} --split {split} "
        + f" --gpu-id {gpu_id} "
    )
    if overwrite:
        cmd += "--overwrite "
    print(f"{Emo.INFO} Running command:")
    print(cmd)
    c.run(cmd)
    print(f"{Emo.GOOD} Model evaluated")


@task
def evaluate_coref(
    c: Context,
    run_name: str,
    split: Literal["dev", "test"] = "dev",
    dataset: Literal["cdt"] = "cdt",
    gpu_id: Optional[int] = None,
):
    """Evaluate the coreference model."""
    echo_header(f"{Emo.DO} Evaluating coreference model")

    training_path = Path("training")
    training_path.mkdir(parents=True, exist_ok=True)
    model_path = training_path / (run_name + ".assembled")

    if gpu_id is None:
        gpu_id = GPU_ID

    cmd = (
        f"{PYTHON} scripts/evaluate_coref.py {model_path} corpus/{dataset}/{split}.spacy"
        + f" --gpu-id {gpu_id}"
    )
    print(f"{Emo.INFO} Running command:")
    print(cmd)
    c.run(cmd)
    print(f"{Emo.GOOD} Coreference model evaluated")


@task
def assemble_coref(c: Context, run_name: str, model_best: bool = False):
    """Assemble the coreference model."""
    # spacy assemble ${vars.config_dir}/coref.cfg training/coref
    echo_header(f"{Emo.DO} Assembling coreference model")

    training_path = Path("training")
    training_path.mkdir(parents=True, exist_ok=True)
    model_path = training_path / f"{run_name}.span_resolver"
    base_model_path = training_path / f"{run_name}"
    write_path = training_path / f"{run_name}.coref"

    if model_best:
        model_path = model_path / "model-best"
        base_model_path = base_model_path / "model-best"
        write_path = write_path / "model-best"
    else:
        model_path = model_path / "model-last"
        base_model_path = base_model_path / "model-last"
        write_path = write_path / "model-last"

    cmd = (
        f"{PYTHON} -m spacy assemble configs/assemble_coref.cfg"
        + f" {write_path}"
        + f" --components.span_resolver.source {model_path}"
        + f" --components.transformer.source {base_model_path}"
        + f" --components.coref.source {base_model_path}"
    )
    c.run(cmd)
    print(f"{Emo.GOOD} Coreference model assembled")


@task
def assemble(c: Context, run_name: str, model_best: bool = False):
    """Assemble the model."""
    # spacy assemble ${vars.config_dir}/coref.cfg training/coref
    echo_header(f"{Emo.DO} Assembling model")

    training_path = Path("training")
    training_path.mkdir(parents=True, exist_ok=True)
    span_resolver = training_path / f"{run_name}.span_resolver"
    main_model = training_path / f"{run_name}"
    write_path = training_path / f"{run_name}.assembled"

    if model_best:
        span_resolver = span_resolver / "model-best"
        main_model = main_model / "model-best"
    else:
        span_resolver = span_resolver / "model-last"
        main_model = main_model / "model-last"

    cmd = (
        f"{PYTHON} -m spacy assemble configs/assemble.cfg"
        + f" {write_path}"
        + f" --components.span_resolver.source {span_resolver}"
        + f" --paths.model_source {main_model}"
    )
    c.run(cmd)
    print(f"{Emo.GOOD} model assembled")


## --- Package and Publish ------------------------------------


@task
def package(c: Context, run_name: str, size: str, overwrite: bool = False):
    """Package the trained model so it can be installed with pip."""
    echo_header(f"{Emo.DO} Packaging model")

    training_path = Path("training")
    package_path = Path("packages")
    metrics = Path("metrics")
    training_path.mkdir(parents=True, exist_ok=True)
    package_path.mkdir(parents=True, exist_ok=True)

    model_path = training_path / run_name
    metrics_json = metrics / run_name / "scores.json"
    if model_path.exists() and (not overwrite):
        print(
            f"{Emo.FAIL} Model already exists to overwrite it please use the -o/--overwrite flag",
        )
        exit(1)

    name = f"{PROJECT}_{size}_trf"
    cmd = f"{PYTHON} -m spacy package {model_path} {package_path} --name {name} --version {VERSION} --force"
    print(f"{Emo.INFO} Running command:")
    print(cmd)
    c.run(cmd)
    cmd = (
        f"{PYTHON} scripts/update_description.py {package_path}/{LANGUAGE}_{name}-{VERSION}/meta.json template_meta.json {size}"
        + f" --metrics-json {metrics_json} "
    )

    if overwrite:
        cmd += " --overwrite "
    print(f"{Emo.INFO} Running command:")
    print(cmd)
    c.run(cmd)
    c.run(f"rm {package_path}/{LANGUAGE}_{name}-{VERSION}/README.md")
    cmd = f"{PYTHON} -m spacy package {model_path} {package_path} --name {name} --version {VERSION} --meta-path template_meta.json --force --build wheel"
    print(f"{Emo.INFO} Running command:")
    print(cmd)
    c.run(cmd)
    c.run("rm template_meta.json")

    # update readme
    repo = package_path / f"{LANGUAGE}_{name}-{VERSION}"
    cmd = f"{PYTHON} scripts/add_readme_metadata.py {repo}"
    print(f"{Emo.INFO} Running command:")
    print(cmd)
    c.run(cmd)
    print(f"{Emo.GOOD} Model packaged")


# @task
# def publish(c: Context, size: str):
#     """Publish package to huggingface model hub. This task is currently turned off as the push to hub spacy isn't sufficient
#     for the desired documentation. Currently this means that the push to hub step is done manually, but in the future this
#     should be automated."""
#     name = f"{LANGUAGE}_{PROJECT}_{size}_trf-{VERSION}"
#     echo_header(f"{Emo.DO} Publishing model")
#     cmd = (
#         f"{PYTHON} -m spacy huggingface-hub push "
#         + f"packages/{name}/dist/{name}-py3-none-any.whl -m 'Update spaCy pipeline to {VERSION}' -o chcaa"
#     )
#     print(f"{Emo.INFO} Running command")
#     print(cmd)
#     c.run(cmd)
#     echo_header(f"{Emo.GOOD} Model pushed to hub")

## --- Workflows ------------------------------------


@task
def workflow_prepare_to_train(c: Context):
    """Runs: `install` &rarr; `fetch-assets` &rarr; `convert` &rarr; `combine`"""
    install(c)
    fetch_assets(c)
    convert(c)
    combine(c)


@task
def workflow_train(
    c: Context,
    model: str = "vesteinn/DanskBERT",
    run_name: str = "dacy",
    config: Optional[str] = None,
    overwrite: bool = False,
    model_best: bool = False,
    gpu_id=None,
):
    """Runs: `create-knowledge-base` &rarr; `train` &rarr; `prep_span_data` &rarr; `train_span_resolver` &rarr; `assemble`"""
    create_knowledge_base(c, model=model)
    train(
        c,
        model=model,
        run_name=run_name,
        overwrite=overwrite,
        gpu_id=gpu_id,
        config=config,
    )
    prep_span_data(c, run_name=run_name, model_best=model_best)
    train_span_resolver(c, run_name=run_name, model_best=model_best, gpu_id=gpu_id)
    assemble(c, run_name=run_name, model_best=model_best)


@task
def workflow_all(
    c: Context,
    model: str = "vesteinn/DanskBERT",
    run_name: str = "dacy",
    size: str = "medium",
    config: Optional[str] = None,
    overwrite: bool = False,
    model_best: bool = False,
):
    """Runs: `workflow_prepare_to_train` &rarr; `workflow_train` &rarr; `evaluate` &rarr; `package`"""

    workflow_prepare_to_train(c)
    workflow_train(
        c,
        model=model,
        run_name=run_name,
        config=config,
        overwrite=overwrite,
        model_best=model_best,
    )
    evaluate(c, run_name=run_name + ".assembled")
    package(c, run_name=run_name + ".assembled", size=size)
