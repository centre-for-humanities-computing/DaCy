<!-- SPACY PROJECT: AUTO-GENERATED DOCS START (do not remove) -->

# 🪐 spaCy Project: Train Danish DaCy NER transformer models on DANSK

This project template lets you train a fine-grained Named-Entity Recognition model on the DANSK dataset containing 18 types annotations. It takes care of downloading the corpus as well as training, evaluating, packaging and releasing the model. The template uses one of more of the transformer models which have been downloaded via Huggingface: 
  - "jonfd/electra-small-nordic" - small,
  - "NbAiLab/nb-roberta-base-scandi" - medium,
  - "chcaa/dfm-encoder-large-v1" - large
  
You can run from yaml file using spacy project run WORKFLOW/COMMAND


## 📋 project.yml

The [`project.yml`](project.yml) defines the data assets required by the
project, as well as the available commands and workflows. For details, see the
[spaCy projects documentation](https://spacy.io/usage/projects).

### ⏯ Commands

The following commands are defined by the project. They
can be executed using [`spacy project run [name]`](https://spacy.io/api/cli#project-run).
Commands are only re-run if their inputs have changed.

| Command | Description |
| --- | --- |
| `login` | Login for wandb and huggingface-cli |
| `setup_gpu` | Installs dependencies and drivers for NVIDIA GPU |
| `fetch_assets` | Downloads DANSK to assets/ |
| `train` | Trains small DaCy model.  |
| `evaluate` | Evaluate the small model on the test.spacy and save the metrics.  |
| `package` | Package the small trained model so it can be installed.  |
| `publish` | Publish small package to huggingface model hub.  |
| `train_all_models` | Trains DaCy models of small, medium and large. |
| `evaluate_all_models` | Evaluate all models on the test.spacy and save the metrics.  |
| `package_all_models` | Package all trained models so they may be installed.  |
| `publish_all_models` | Publish all model packages to huggingface model hub. |
| `generate_readme` | Auto-generates a README.md with a project description. |
| `clean` | Remove intermediate files |

### ⏭ Workflows

The following workflows are defined by the project. They
can be executed using [`spacy project run [name]`](https://spacy.io/api/cli#project-run)
and will run the specified commands in order. Commands are only re-run if their
inputs have changed.

| Workflow | Steps |
| --- | --- |
| `prepare_data` | `fetch_assets` |
| `train_eval_pack_publ` | `train` &rarr; `evaluate` &rarr; `package` &rarr; `publish` |
| `all_models_train_eval_pack_publ` | `train_all_models` &rarr; `evaluate_all_models` &rarr; `package_all_models` &rarr; `publish_all_models` |

### 🗂 Assets

The following assets are defined by the project. They can
be fetched by running [`spacy project assets`](https://spacy.io/api/cli#project-assets)
in the project directory.

| File | Source | Description |
| --- | --- | --- |
| [`assets/dansk.spacy`](assets/dansk.spacy) | Local | The full to-be-published DANSK dataset |

<!-- SPACY PROJECT: AUTO-GENERATED DOCS END (do not remove) -->

### 🚧 Requirements and setup

The models were trained on a Virtual Machine that was accessed through UCloud, Aalborg University. 
The system was based on Cuda + Jupyter on Ubuntu v. 20.04, using a uc-t4-1 / uc-t4gpu, and was setup using the specifications in `server_setup`

The environment as well as requirements for the Python setup can be automatically installed using the `environment_setup.sh`, which is partially based on the dependencies in `requirements.txt`

### 📜 Data split

The DANSK dataset split for this project contains following elements:

| File | Source | Description |
| --- | --- | --- |
| [`corpus/train.spacy`](corpus/train.spacy) | Local | The training partition of the full DANSK dataset |
| [`corpus/dev.spacy`](corpus/dev.spacy) | Local | The dev partition of the full DANSK dataset |
| [`corpus/test.spacy`](corpus/test.spacy) | Local | The testing partition of the full DANSK dataset |

The distribution of documents and entities for the partitions can be seen in the table below:

|                   | **Full** |    **Train**   |    **Dev**   |    **Test**   |
|:-----------------:|:--------:|:--------------:|:------------:|:-------------:|
|     **#Docs**     |   15062  |   12062 (80%)  |  1500 (10%)  |   1500 (10%)  |
|     **#Ents**     |   14462  | 11638 (80.47%) | 1327 (9.18%) | 1497 (10.25%) |
|   **#CARDINAL**   |   2069   |  1702 (82.26%) |  168 (8.12%) |  226 (10.92%) |
|     **#DATE**     |   1756   |  1411 (80.35%) | 182 (10.36%) |  163 (9.28%)  |
|     **#EVENT**    |    211   |  175 (82.94%)  |  19 (9.00%)  |   17 (8.06%)  |
|   **#FACILITY**   |    246   |  200 (81.30%)  |  25 (10.16%) |   21 (8.54%)  |
|      **#GPE**     |   1604   |  1276 (79.55%) |  135 (8.42%) |  193 (12.03%) |
|   **#LANGUAGE**   |    126   |   53 (42.06%)  |  17 (13.49%) |  56 (44.44%)  |
|      **#LAW**     |    183   |  148 (80.87%)  |  17 (9.29%)  |   18 (9.84%)  |
|   **#LOCATION**   |    424   |  351 (82.78%)  |  46 (10.85%) |   27 (6.37%)  |
|     **#MONEY**    |    714   |  566 (79.27%)  |  72 (10.08%) |  76 (10.64%)  |
|     **#NORP**     |    495   |  405 (81.82%)  |  41 (8.28%)  |   49 (9.90%)  |
|    **#ORDINAL**   |    127   |  105 (82.68%)  |  11 (8.66%)  |   11 (8.66%)  |
| **#ORGANIZATION** |   2507   |  1960 (78.18%) |  249 (9.93%) |  298 (11.87%) |
|    **#PERCENT**   |    148   |  123 (83.11%)  |  13 (8.78%)  |   12 (8.11%)  |
|    **#PERSON**    |   2133   |  1767 (82.84%) |  191 (8.95%) |  175 (8.20%)  |
|    **#PRODUCT**   |    763   |  634 (83.09%)  |  57 (7.47%)  |   72 (9.44%)  |
|   **#QUANTITY**   |    292   |  242 (82.88%)  |  28 (9.59%)  |   22 (7.53%)  |
|     **#TIME**     |    218   |  185 (84.86%)  |  18 (8.26%)  |   15 (6.88%)  |
|  **#WORK OF ART** |    419   |  335 (79.95%)  |  38 (9.07%)  |  46 (10.98%)  |
