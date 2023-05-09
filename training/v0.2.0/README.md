


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


# 📋 tasks.py

| Task | Description |
| --- | --- |
| `create_readme` | Creates a readme file with the project workflow from the tasks.py file |
| `create_venv` | Create a virtual environment. Optional. |
| `check_gpu` | Check if spacy gpu support is working |
| `install` | Install the project and logs in to wandb |
| `fetch_assets` | Fetch assets for model training. |
| `convert` | Convert the data to the correct format |
| `combine` | Combine the data CDT and DDT datasets |
| `create_knowledge_base` | Create the Knowledge Base in spaCy and write it to file. |
| `train` | train a model using spacy train Args: embedding_size: The size of the transformer embedding. If None the default size is used. |
| `further_train` | train a model using spacy train Args: embedding_size: The size of the transformer embedding. If None the default size is used. |
| `prep_span_data` | Prepare data for the span resolver component. |
| `train_span_resolver` | Train the span resolver component. |
| `evaluate` | Evaluate a model using spacy evaluate |
| `evaluate_coref` | Evaluate the coreference model. |
| `assemble_coref` | Assemble the coreference model. |
| `workflow_prepare_to_train` | Runs: `install` &rarr; `fetch-assets` &rarr; `convert` &rarr; `combine` &rarr; `create-knowledge-base` |
| `workflow_train_` | Runs: `train` &rarr; `prep_span_data` &rarr; `train_span_resolver` &rarr; `train_further` &rarr; `assemble` |

    