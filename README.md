
<div align="center"><img src="img/icon.png" height="200px"/></div>

<h1 align="center">DaCy: A SpaCy NLP Pipeline for Danish</h1>

[![release version versions](https://img.shields.io/badge/DaCy%20Version-0.0.0-green)](https://github.com/KennethEnevoldsen/DaCy)
[![python versions](https://img.shields.io/badge/Python-%3E=3.6-blue)](https://github.com/KennethEnevoldsen/DaCy)
<!-- [![Github All Releases](https://img.shields.io/github/downloads/kennethenevoldsen/dacy/total.svg)]() -->
[![Code style: black](https://img.shields.io/badge/Code%20Style-Black-black)](https://black.readthedocs.io/en/stable/the_black_code_style.html)
[![license](https://img.shields.io/github/license/KennethEnevoldsen/DaCy.svg?color=blue)](https://github.com/KennethEnevoldsen/DaCy)

DaCy is a Danish preprocessing pipeline trained in SpaCy. At the time of writing it has achieved State-of-the-Art performance on all Benchmark tasks for Danish. This repository contains code for reproducing DaCy as well as download and loading the models. Furthermore it also contains guides on how to use DaCy.

# 🔧 Installation
it currently only possible to download DaCy directly from GitHub, however this can be done quite easily using:
```bash
pip install git+https://github.com/KennethEnevoldsen/DaCy
```

# 👩‍💻 Usage
To use the model you first have to download either the `medium` or `large` model. To see a list of all available models:

```python
import dacy
for model in dacy.models():
    print(model)
# da_dacy_small_tft-0.0.0
# da_dacy_medium_tft-0.0.0
# da_dacy_large_tft-0.0.0
```

To download and load a model simply execute:
```python
nlp = dacy.load("da_dacy_medium_tft-0.0.0")
```

Which will download the model to the `.dacy` directory in your home directory. To figure out where this you can always use:

```python
where_is_my_dacy()
```
which for __me__ will return `'/Users/kenneth/.dacy'`. 

To download the model to a specific directory:
```python
dacy.download_model("da_dacy_medium_tft-0.0.0", your_save_path)
nlp = dacy.load_model("da_dacy_medium_tft-0.0.0", your_save_path)
```

DaCy also include a Jupyter notebook tutorial. If you do not have Jupyter Notebook installed, instructions for installing and running it can be found [here]( http://jupyter.org/install). All the tutorial are located in the `tutorials` folder.

| Tutorial                                                                                                | Content                                                                                                              | file name                 |                                                                                                                                                                                             |
| ------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- | ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [Introduction](https://github.com/KennethEnevoldsen/DaCy/blob/main/tutorials/dacy-spacy-tutorial.ipynb) | A simple introduction to SpaCy and DaCy. For a more detailed instruction I recommend the course by SpaCy themselves. | dacy-spacy-tutorial.ipynb | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KennethEnevoldsen/DaCy/blob/main/tutorials/dacy-spacy-tutorial.ipynb) |
| [Sentiment](https://github.com/KennethEnevoldsen/DaCy/blob/main/tutorials/dacy-sentiment.ipynb)         | A simple introduction to the new sentiment features in DaCy.                                                         | dacy-sentiment.ipynb      | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KennethEnevoldsen/DaCy/blob/main/tutorials/dacy-sentiment.ipynb)      |
| [wrapping a fine-tuned Tranformer](https://github.com/KennethEnevoldsen/DaCy/blob/main/tutorials/dacy-wrapping-a-classification-transformer.ipynb)         | A guide on how to wrap an already fine-tuned transformer to and add it to your SpaCy pipeline using DaCy helper functions.                                                         | dacy-wrapping-a-classification-transformer.ipynb      | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KennethEnevoldsen/DaCy/blob/main/tutorials/dacy-wrapping-a-classification-transformer.ipynb)      |




<!--
### Lemmatization
To obtains SOTA performance in lemmatization as well you should add [this lemmatization](https://github.com/sorenlind/lemmy) pipeline as well:

```python
import lemmy.pipe

pipe = lemmy.pipe.load('da')

# Add the component to the spaCy pipeline.
nlp.add_pipe(pipe, after='tagger')

# Lemmas can now be accessed using the `._.lemmas` attribute on the tokens.
nlp("akvariernes")[0]._.lemmas
```

This requires you install the package beforehand, this is done easily using:

```
pip install lemmy
```

Note: that pipeline haven't yet been adapted to spacy v3 as of the time of writing.
-->


# 🦾 Performance and Training

The following table show the performance on DaNE when compared to other models. Highest scores are highlighted with **bold** and second highest is <ins>underlined</ins>
<div align="center"><img src="img/perf.png"/></div>

Want to learn more about how the model was trained, check out this [blog post](https://www.kennethenevoldsen.com/post/new-fast-and-efficient-state-of-the-art-in-danish-nlp/).

## Reproduction
the folder `DaCy` contains a SpaCy project which will allow for a reproduction of the results. This folder also includes the evaluation metrics on DaNE.

For further instructions on this look up the project file `DaCy_training/project.yml`.


# 🤔 Issues and Usage Q&A

To ask questions, report issues or request features, please use the [GitHub Issue Tracker](https://github.com/KennethEnevoldsen/DaCy/issues). Question related to SpaCy is kindly referred to the SpaCy GitHub or forum.

## Acknowledgements
This is really an acknowledgement of great open-source software and contributors. This wouldn't have been possible with the work by the SpaCy team which developed an integrated the software. Huggingface for developing Transformers and making model sharing convenient. BotXO for training and sharing the Danish BERT model and Malte Bertelsen for making it easily available. DaNLP has made it extremely easy to get access to Danish resources to train on and even supplied some of the tagged data themselves and have done great job of developing these datasets.

## References

If you use this library in your research, please kindly cite:

```bibtex
@inproceedings{enevoldsen2020dacy,
    title={DaCy: A SpaCy NLP Pipeline for Danish},
    author={Enevoldsen, Kenneth},
    year={2021}
}
```

## License

DaCy is released under the Apache License, Version 2.0. See the `LICENSE` file for more details.
