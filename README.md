
<div align="center"><img src="img/icon.png" height="200px"/></div>

<h1 align="center">DaCy: A SpaCy NLP Pipeline for Danish</h1>

[![release version versions](https://img.shields.io/badge/DaCy%20Version-0.0.0-blue)](https://github.com/KennethEnevoldsen/DaCy)
[![python versions](https://img.shields.io/badge/Python-%3E=3.6-blue)](https://github.com/KennethEnevoldsen/DaCy)
[![python versions](https://img.shields.io/badge/SpaCy-%3E=3.0.0-blue)](https://github.com/KennethEnevoldsen/DaCy)
[![Code style: flake8](https://img.shields.io/badge/Code%20Style-flake8-greem)](https://pypi.org/project/flake8/)
[![DaCy: Download](https://img.shields.io/badge/Download%20Status-online-green)](https://sciencedata.dk/shared/0e5d0b97fbead07d1f2ba7c3cbea03eb)


DaCy is a Danish preprocessing pipeline trained in SpaCy. At the time of writing it has achieved State-of-the-Art performance on all Benchmark tasks for Danish. This repository contains code for reproducing DaCy. To download the models use the DaNLP package (request pending), SpaCy ([request pending](https://github.com/explosion/spaCy/issues/7221)) or downloading the project directly [here](https://sciencedata.dk/shared/0e5d0b97fbead07d1f2ba7c3cbea03eb).

## Reproduction
the folder `DaCy` contains a SpaCy project which will allow for a reproduction of the results. This folder also includes the evaluation metrics on DaNE.

For further instructions on this look up the project file `DaCy/project.yml`.

## Usage

To load in the project using the direct download simple place the downloaded "packages" folder in your directory load the model using SpaCy:

```python
import spacy
nlp = spacy.load("da_dacy_large_tft-0.0.0")
```

More explicitly from the unpacked folder it is:
```
nlp = spacy.load("da_dacy_large_tft-0.0.0/da_dacy_large_tft/da_dacy_large_tft-0.0.0")
```
Thus if you get an error you might be loading from the outer folder called `da_dacy_large_tft-0.0.0` rather than the inner.

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

## Performance and Training

The following table show the performance on DaNE when compared to other models. Highest scores are highlighted with **bold** and second highest is <ins>underlined</ins>
<div align="center"><img src="img/perf.png"/></div>

Want to learn more about how the model was trained, check out this [blog post](https://www.kennethenevoldsen.com/post/new-fast-and-efficient-state-of-the-art-in-danish-nlp/).

## Issues and Usage Q&A

To ask questions, report issues or request features 🤔 , please use the [GitHub Issue Tracker](https://github.com/KennethEnevoldsen/DaCy/issues). Question related to SpaCy is referred to the SpaCy GitHub or forum.


### Acknowledgements
This is really an acknowledgement of great open-source software and contributors. This wouldn't have been possible with the work by the SpaCy team which developed an integrated the software. Huggingface for developing Transformers and making model sharing convenient. BotXO for training and sharing the Danish BERT model and Malte Bertelsen for making it easily available. DaNLP has made it extremely easy to get access to Danish resources to train on and even supplied some of the tagged data themselves and does a great job of actually developing these datasets.

### References

If you use this library in your research, please kindly cite:

```bibtex
@inproceedings{enevoldsen2020dacy,
    title={DaCy: A SpaCy NLP Pipeline for Danish},
    author={Enevoldsen, Kenneth},
    year={2021}
}
```

## LICENSE

DaCy is released under the Apache License, Version 2.0. See the `LICENSE` file for more details.
