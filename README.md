# diachrony_for_russian
This repository contains the models and the dataset related to the paper "Tracing cultural diachronic semantic shifts in Russian
using word embeddings: test sets and baselines" by Vadim Fomin, Daria Bakshandayeva, Julia Rodina and Andrey Kutuzov, accepted for the conference Dialog 2019.

# datasets

The "datasets" directory contains the dataset annotated.csv discussed in the mentioned paper.
It consists of 280 entries. For each entry, three annotators considered a word from the column WORD (say свиной 'of a swine, related to a swine') and decided how much the word in question has changed its meaning from year in the column BASE_YEAR (e. g. 2009, the year when swine flu was widely discussed in media) to the next year. The assessments of each annotator are contained in the columns ASSESSOR1, ASSESSOR2, and ASSESSOR3. The annotator's assessment was on the scale from 0 to 2; the arithmetic mean of the scores was counted (column ASSESSOR_MEAN) and rounded to the nearest integer. The rounded value was considered to be the ground truth (column GROUND TRUTH.)

The words for the dataset were sampled as follows: a Global Anchors model was asked to produce top 10 changed words for 2001 as compared to 2000, ..., 2014 as compared to 2013, producing 140 words that may have experienced a semantic shift during one of those years. For each of those words, we then picked a filler from a the same frequency decile and added it to the dataset. Words that were picked as fillers contain "0" in the column "LABEL"; others contain "1". Words with LABEl=1 also contain a value in column RATING; e. g. the first word from a top-10 has a rating of 1 and the tenth word has a rating of 10.

# models

The "models" directory contains the models that we used to trace semantic shifts in Russian words, namely the procrustes alignment model, the global anchors model, the Kendall tau model and the Jaccard measire model.

# using the models

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
