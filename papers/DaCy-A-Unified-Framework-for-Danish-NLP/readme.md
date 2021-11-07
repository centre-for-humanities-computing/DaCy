# DaCy: A Unified Framework for Danish NLP

## Abstract

Danish natural language processing (NLP) has in recent years obtained considerable improvements with the addition of multiple new datasets and models. However, at present, there is no coherent framework for applying state-of-the-art models for Danish. We present DaCy: a unified framework for Danish NLP built on SpaCy. DaCy uses efficient multitask models which obtain state-of-the-art performance on named entity recognition, part-of-speech tagging, and dependency parsing. DaCy contains tools for easy integration of existing models such as for polarity, emotion, or subjectivity detection. In addition, we conduct a series of tests for biases and robustness of Danish NLP pipelines through augmentation of the test set of DaNE. DaCy large compares favourably and is especially robust to long input lengths and spelling variations and errors. All models except DaCy large display significant biases based on ethnicity while only Polyglot show a significant bias based on gender. We argue that for languages with limited benchmark sets, data augmentation can be particularly useful for obtaining more realistic and fine-grained performance estimates. We provide a series of augmenters as a first step towards a more thorough evaluation of language models for low and medium resource languages and encourage further development.

## Reading the paper

Read the paper on arXiv [here](https://arxiv.org/abs/2107.05295).

## Reproducing the Paper

To reproduce the paper you will first need to install the required packages. Notice that this require Python version 3.7.2 or higher. We suggest you make a virtual environment before doing this.

```
pip install -r requirements.txt
```

Do note that as we are testing a wide variety of packages not all package versions are compatible. Furthermore, to test Polyglot you will need to install additional dependencies outside of Python. You can follow the install instructions on their [website](https://polyglot.readthedocs.io/en/latest/Installation.html) for this. If you want to test the performance on GPU you will also need to [install](https://spacy.io/usage) the compatible cupy and spacy version for your CUDA driver.

After that simply running:

```
python dacy_paper_replication.py
```

Will generate a series of `.csv`s in `robustness` folder, under the name `robustness/{mdl}_augmentation_performance.csv`.

## Getting the results
For convenience, we have included our version of the combined dataset of above mentioned `.csv`s which was used to reproduce the tables in the paper.

## Creating the tables
The R markdown `robustness.Rmd` contains the code for automatically generating the LateX tables.
