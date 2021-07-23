# DaCy Training
This folder contains the resources for training DaCy models. Each folder contain code for training each iteration of DaCy models, where the version number corresponds to the DaCy version. The `configs` folder includes the configs used for all trained models. While the `project.yml` include the workflows used for training the models, with each model containing its own workflow. These workflows and their subcommands can be called using `spacy project run [WORKFLOW/COMMAND]`. For example in `v0.1.0` if you wish to train the small model you would run:

```
spacy project run small
```


The `requirements.txt` includes the requirements for running the projects, note that these are installed using the workflow in the `project.yml` so you will not need to install these manually (with the exception of spacy).


# Version 0.1.1 (experimental)
This is a version of DaCy trained with a series of augmentation, hopefully to improve performance of the model on downstream tasks. Comparing these model with the models from 0.1.0 we draw the following conclusions:

- DaCy small performance:
  - w. augmentation (dacy-v1):
    - similar performance as without augmentation
    - better peformance on muslim names
    - way better performance on keystroke errors
    - better handling of abbreviations
    - notably worse performance on casing
- Dacy medium performance:
    - w. augmentation (dacy-v1):
        - no augmentation: 1 pp better performance on NER with augmentation and minor on the rest
        - otherwise very similar performance
- DaCy large performance:
    - w. augmentation (dacy-v1):
        - no augmentation: similar or slightly better without aug.
        - 8 pp lower on lowercasing
        - 2-5-10 pp. (2%, 5%, 15%)  improvement in performance when dealing with keystroke errors
        - slightly worse performance on muslim names


# Version 0.1.0
This is an update of DaCy using SpaCy v3.1.0 for compatibility. The primary changes here include:

- Minor changes to the training procedure (e.g. smaller hidden width in the NER and a higher value for patience)
- Tokenization no longer strips accents, leading to a meaningful difference between e.g. a and å.
- Inclusion of a the components `lemmatizer`, `morphologizer`, and `attribute_ruler`

for the exact differences we recommend you examine the `config.cfg` files.

# Version 0.0.0
This was the first version on DaCy

## Performance
The following table shows the performance on the DaNE dataset of models trained for DaCy. Highest scores are highlighted with **bold** and the second highest is <ins>underlined</ins>. The models which do not have a DaCy name is not included in DaCy as faster and better performing models were available. E.g. while the ConvBert model are fast their don't compare favourably to the Ælæctra Model.

<div align="center"><img src="img/perf_training.png"/></div>

Where the following models are trained by:

- XLM-Roberta-Large: Facebook
- DaConvBERT: Philip Tamimi-Sarnikowski
- DaELECTRA: Philip Tamimi-Sarnikowski
- Ælæctra (cased and uncased): Malte Højmark-Bertelsen
- DaBERT: BotXO (supplied on Huggingface model hub by Malte Højmark-Bertelsen)

# Training Report

For reproducibility it is possible to view a report on the training of models in DaCy on [here](https://wandb.ai/kenevoldsen/dacy-an-efficient-pipeline-for-danish/reports/DaCy-Training-performances--Vmlldzo1NDgyNzk?accessToken=bavawchq2sfno773xne0texhk5ni6mh018ft3ghxg5la36tn7xr91mxapq4lshec).
