
<div align="center"><img src="img/icon.png" height="200px"/></div>

<h1 align="center">DaCy: A SpaCy NLP Pipeline for Danish</h1>

DaCy is a Danish preprocessing pipeline trained in SpaCy. At the time of writing it have achieved State-of-the-Art performance on all Benchmark tasks for Danish. This repository contains code for reproducing DaCy. To download the model use the DaNLP package (request pending), SpaCy (request pending) or downloading the project directly ([link coming](missing)).

## Reproduction
the folder `DaCy` contains a SpaCy project which will allow for a reproduction of the results. This folder also include the evaluation metrics on DaNE.

## Usage

To load in the project using the direct download simple place the downloaded "packages" folder in your directory load the model using SpaCy:

```python
import spacy
spacy.load("packages/da_dacy_large_tft-0.0.0")
```

## Performance

The following table show the performance on DaNE when compared to other models. Highest scores are highlighted with **bold** and second highest is <ins>underlined</ins>
<div align="center"><img src="img/perf.png"/></div>


## Issues and Usage Q&A

To ask questions, report issues or request features 🤔 , please use the [GitHub Issue Tracker](https://github.com/KennethEnevoldsen/DaCy/issues). Question related to SpaCy are refered to the SpaCy github or forum.


## References

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