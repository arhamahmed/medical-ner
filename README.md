# medical-ner

The code in this repository aims to reproduce the results from the paper `Fully‐connected LSTM–CRF on medical concept extraction` [1].

The goal is medical concept extraction (an NER task) from clinical notes through use of different LSTM models, notably a "fully-connected" LSTM structure from the paper. Input to the system are two sets of files: raw clinical text documents and concept files in some form of CoNLL format that serve as labels for the corresponding clinical outputs. The model tags each word to a set of tags (problem, treatment, test) using an IOB (inside, outside, beginning) scheme. For example

| The | patients| irregular | heartbeats | require | a | medical | resonance | imaging | test | every | week | to | collect | data | points |
| - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| O | O| B-problem | I-problem | O | a | B-test | I-test | I-test | I-test | O | O | O | O | O | O |

## Setup
The requirements below were run on a Windows OS with Python 3.10.

- run: pip install -qr requirements.txt
- install `graphviz` from this [link](https://graphviz.org/download/)
- download pretrained GloVe embeddings (Wikipedia 2014 and/or CommonCrawl 840B) from [here](https://nlp.stanford.edu/projects/glove/); place in data/ accordingly
- get the i2b2 dataset from the Data Acquisition section; place in the `data` folder after extracting
- run `py process_data.py` to perform data pre-processing and formatting
- run `py main.py` and follow the single prompt on which LSTM structure to use (1 = UniLSTM, 2 = BiLSTM, 3 = LSTM w/attention, 4 = 4-layer Stacked LSTM, 5 = FC-LSTM)

The final command from the above loads the data, GloVe vectors, builds/trains a model with the selected LSTM structure then performs predictions on the test dataset and finally outputs precision/recall/F1 scores.

### Data Acquisition

The paper makes use of the 2010 i2b2 Challenge datasets which as been consolidated and [renamed](https://www.i2b2.org/NLP/DataSets/Main.php) to the n2c2 dataset. It is available on the [DBMI Portal](https://portal.dbmi.hms.harvard.edu) and requires signing of a data-use agreement. Once granted access, download the training and test data. In each folder, create a `txt` and `concept` folder and appropriately place all the text/concept files in them; while there may be multiple partners/sets of training/test data the folder structure should look like the following:

* data
    * training_data
        * txt: contains all clinical note txt files
        * concept: contains CoNLL formatted concept files
    * test
        * txt: contains all clinical note txt files
        * concept: contains CoNLL formatted concept files

## Results
(using the default/fixed seed `123`)

| Structure | Precision | Recall | F1-score |
| -- | -- | -- | -- |
| UniLSTM (paper) | 79.95 | 75.14 | 77.38 |
| UniLSTM (current implementation) | 78.78 | 71.62 | 75.00 |
| BiLSTM (paper) | 84.59 | 82.93 | 83.75 |
| BiLSTM (current implementation) | 82.93 | 73.06 | 77.62 |
| Attention LSTM (paper) | 84.39 | 83.25 | 83.82 |
| Attention LSTM (current implementation) | 78.11 | 73.60 | 75.76 |
| Stacked BiLSTM (paper) | 84.43 | 83.40 | 83.91 |
| Stacked BiLSTM (current implementation) | 81.86 | 76.75 | 79.21 |
| FC BiLSTM (paper) | 85.09 | 83.23 | 84.15 |
| FC BiLSTM (current implementation) | 81.59 | 74.20 | 77.67 |

## References

[1] Ji, Jie & Chen, Bairui & Jiang, Hongcheng. (2020). Fully-connected LSTM–CRF on medical concept extraction. International Journal of Machine Learning and Cybernetics. 11. 10.1007/s13042-020-01087-6. 