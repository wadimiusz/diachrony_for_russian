# diachrony_for_russian
This repository contains the models and the dataset related to detecting semantic changes.

The "datasets" directory contains the dataset annotated.csv discussed in the mentioned paper.

The "models" directory contains the models that we used to trace semantic shifts in Russian words, namely the procrustes alignment model, the global anchors model, the Kendall tau model and the Jaccard measire model.

If you have two models, 2000.model and 2014.model, trained upon texts from 2000 and 2014 respectively, and you wish to evaluate how much semantic change the word "несогласный" has experienced from 2000 to 2014, you can run the script score_word as follows:
```
python3 score_word.py -w несогласный -m 2000.model -m 2014.model
```
This will print in stdout the scores according to each of the 4 models:

```
KendallTau score: -0.05795918367346939 (from -1 to 1)
Jaccard score: 0.0 (from 0 to 1)
Global Anchors score: 0.36681556701660156 (from -1 to 1)
Procrustes aligner score: 0.17986169457435608 (from -1 to 1)
```
