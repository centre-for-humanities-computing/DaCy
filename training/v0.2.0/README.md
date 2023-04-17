


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

- [ ] Corefs and NED are only available for a subset of the corpus? Would it be better to train these independently?
- [ ] DaWikiNED is not currently used, but could be used to improve the NED model. In the [paper](https://aclanthology.org/2021.crac-1.7.pdf) 
it only improved the model from 0.85 to 0.86 so it might not be worth it.
- [ ] Can the entity linker model use non-entity QIDs? We have quite a few of these in the DaNED dataset.
- [ ] DANSK is currently not included. It could be added
- [ ] It would be interested to see if anything could be gained from using a multilingual approach e.g. include the english ontonotes
or Norwegian Bokmål.

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
| `install` | Install the project and logs in to wandb |
| `fetch_assets` | Fetch assets for model training |
| `convert` | Convert the data to the correct format |
| `combine` | Combine the data CDT and DDT datasets |
| `train` | train a model using spacy train |
| `evaluate` | Evaluate a model using spacy evaluate |
| `workflow_prepare_to_train` | Runs: `install` &rarr; `fetch-assets` &rarr; `convert` &rarr; `combine` |

    