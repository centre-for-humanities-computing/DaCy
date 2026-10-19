<!-- WEASEL: AUTO-GENERATED DOCS START (do not remove) -->

# 🪐 Weasel Project: DaCy: NLP for Danish

Train, evaluate, and package DaCy v0.2.9 models.  

Built on:  

| Dataset | Source | Adds | Unit |
|---|---|---|---|
| [**DDT**](https://github.com/UniversalDependencies/UD_Danish-DDT) | UniversalDependencies | POS, morphology, lemmas, dependencies | sentences |
| [**DaNE**](https://danlp-alexandra.readthedocs.io/en/latest/docs/datasets.html#dane) | Alexandra Institute | Named entities on DDT | same sentences as DDT |
| [**CDT (DaCoref)**](https://danlp-alexandra.readthedocs.io/en/latest/docs/datasets.html#dacoref) | Alexandra Institute | Coreference, document boundaries | documents, some sentences overlap with DDT |

**Building the corpus**
1. DDT and DaNE are converted to `.spacy`.
2. CDT documents are reassigned to splits compatible with DDT.
   (`create_ddt_compatible_splits_for_cdt.py`).
3. `combine.py` matches CDT and DDT/DaNE on `sent_id`. It writes CDT documents to `corpus/cdt/` and DDT sentences with no CDT document to `corpus/cdt_ddt/data.spacy`.
4. `merge_ddt_sents_with_cdt.py` adds those single-sentence docs to the CDT splits giving `corpus/cdt_ddt/{train,dev,test}.spacy`.

Run everything with `spacy project run all`. 

`project.yml` also has a function to `evaluate_llm` to evaluate LLM-based components from [spacy-llm](https://spacy.io/usage/large-language-models).

Note on data: 
- DDT and DaNE share the same sentences. 
- DaNE adds NER annotations to some sentences in DDT.
- CDT is a separate treebank with its own texts. But some CDT sentences overlap with DDT sentences (sent_id matching works for those).
- "remaining docs" in combine.py are DDT sentences that have no corresponding CDT document.


## 📋 project.yml

The [`project.yml`](project.yml) defines the data assets required by the
project, as well as the available commands and workflows. For details, see the
[Weasel documentation](https://github.com/explosion/weasel).

### ⏯ Commands

The following commands are defined by the project. They
can be executed using [`weasel run [name]`](https://github.com/explosion/weasel/tree/main/docs/cli.md#rocket-run).
Commands are only re-run if their inputs have changed.

| Command | Description |
| --- | --- |
| `verify_gpu` | Check if GPU is available to spaCy and CuPy |
| `install_vectors_lg` | Installs pretrained vectors from assets/da_core_news_lg for DaCy tiny |
| `prepare_dane` | Prepare DaNE data |
| `prepare_dacoref` | Prepare dacoref (CDT) data |
| `preprocess_ddt` | Convert the DDT data to `.spacy` |
| `preprocess_dane` | Convert the DaNE data to `.spacy` |
| `combine` | Combine CDT and DDT data |
| `merge_cdt_ddt` | Merge CDT data with single-sentence DDT data |
| `train` | Train dacy_large on cdt_ddt data |
| `evaluate` | Evaluate trained pipeline and save the metrics |
| `evaluate_llm` | Evaluate LLM-based pipeline and save the metrics |
| `package` | Package trained model (NOTE: uses `--build wheel` to override `sdist` default as per [HuggingFace recommendations](https://huggingface.co/docs/hub/main/en/spacy#using-the-spacy-cli-recommended)) |
| `clean` | Remove intermediate files |

### ⏭ Workflows

The following workflows are defined by the project. They
can be executed using [`weasel run [name]`](https://github.com/explosion/weasel/tree/main/docs/cli.md#rocket-run)
and will run the specified commands in order. Commands are only re-run if their
inputs have changed.

| Workflow | Steps |
| --- | --- |
| `all` | `install_vectors_lg` &rarr; `prepare_dane` &rarr; `prepare_dacoref` &rarr; `preprocess_dane` &rarr; `preprocess_ddt` &rarr; `combine` &rarr; `merge_cdt_ddt` &rarr; `train` &rarr; `evaluate` &rarr; `package` |

### 🗂 Assets

The following assets are defined by the project. They can
be fetched by running [`weasel assets`](https://github.com/explosion/weasel/tree/main/docs/cli.md#open_file_folder-assets)
in the project directory.

| File | Source | Description |
| --- | --- | --- |
| `assets/UD_Danish-DDT` | Git |  |
| `assets/dane.zip` | URL |  |
| `assets/dacoref.zip` | URL |  |
| `assets/da_core_news_lg-3.8.0-py3-none-any.whl` | URL |  |

<!-- WEASEL: AUTO-GENERATED DOCS END (do not remove) -->