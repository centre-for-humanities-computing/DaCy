


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


# 📋 tasks.py

| Task | Description |
| --- | --- |
| `create_readme` | Creates a readme file with the project workflow from the tasks.py file |
| `check_gpu` | Check if spacy gpu support is working |
| `install` | Install the project and logs in to wandb |
| `fetch_assets` | Fetch assets for model training |
| `convert` | Convert the data to the correct format |
| `combine` | Combine the data CDT and DDT datasets |
| `train` | train a model using spacy train |
| `evaluate` | Evaluate a model using spacy evaluate |
| `train_coref_cluster` | Train the clustering component |
| `prep_span_data` | Prepare data for the span resolver component. |
| `train_span_resolver` | Train the span resolver component. |
| `assemple_coref` | Assemble the coreference model. |
| `workflow_prepare_to_train` | Runs: `install` &rarr; `fetch-assets` &rarr; `convert` &rarr; `combine` &rarr; `create-knowledge-base` |
| `workflow_train_coref_model` | Runs: `train_coref_cluster` &rarr; `prep_span_data` &rarr; `train_span_resolver` &rarr; `assemple_coref` |
| `create_knowledge_base` | Create the Knowledge Base in spaCy and write it to file. |
| `train_ned` | Train the named entity disambiguation component. |
| `evaluate_ned` | Evaluate the named entity disambiguation component. |
| `workflow_train_ned` | Runs: `create_knowledge_base` &rarr; `train_ned` &rarr; `evaluate_ned` |

    