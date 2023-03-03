<!-- SPACY PROJECT: AUTO-GENERATED DOCS START (do not remove) -->

# 🪐 spaCy Project: Train Danish DaCy NER transformer models on DANSK

This project template lets you train a fine-grained Named-Entity Recognition model on the DANSK dataset containing 18 types annotations. It takes care of downloading the corpus as well as training, evaluating, packaging and releasing the model. The template uses one of more of the transformer models which have been downloaded via Huggingface: 
  - "jonfd/electra-small-nordic"
  - "NbAiLab/nb-roberta-base-scandi", 
  - "KennethEnevoldsen/dfm-bert-large-v1-2048bsz-1Msteps"
  
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
| `fetch_assets` | Downloads DANSK to assets/ |
| `split_dansk` | Splits DANSK into train, dev, test |
| `train` | Trains test DaCy model |
| `evaluate` | Evaluate the test model on the test.spacy and save the metrics |
| `package` | Package the test trained model so it can be installed |
| `publish` | Publish test package to huggingface model hub. |
| `train_all_models` | Trains DaCy models of small, medium and large |
| `evaluate_all_models` | Evaluate all models on the test.spacy and save the metrics |
| `package_all_models` | Package all trained models so they may be installed |
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
| `prepare_data` | `fetch_assets` &rarr; `split_dansk` |
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