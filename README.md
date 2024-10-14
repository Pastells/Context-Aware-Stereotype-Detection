#  Context-Aware Stereotype Detection Conversational Thread Analysis on BERT-based Models [SEPLN 2024]


This repository contains the code for classification models of racial stereotypes in Spanish corpora (StereoHoax-ES and DETESTS), taking into account their conversational contexts.

Both corpora are available in [HuggingFace](https://huggingface.co/datasets/CLiC-UB/DETESTS-Dis).

We consider different contexts for the text classification:

|         | StereoHoax (Tweet)            | DETESTS (Sentence) |
| ------- | ----------------------------- | ------------------ |
| Level 1 | None                          | Previous sentences |
| Level 2 | Parent tweet (text father)    | Previous comment   |
| Level 3 | Conversation head (text head) | First comment      |
| Level 4 | Racial hoax (rh text)         | News title         |

For both corpora, we fill the missing contexts with the ones one level above.
E.i. if the "parent tweet" is missing, we use the "conversation head". In case
it is also missing, we use the "racial hoax", which is always available. For
DETESTS, the "news title" is also always available.

**Disclaimer**: the original repository was used with more corpora and options than the ones presented on the article above.
For example:

- Task 1 refers to stereotype prediction (yes/no) [**this paper**].
- Task 2 was the stereotype impliciteness classification (imp/exp), see [DETESTS-Dis: DETEction and classification of racial STereotypes in Spanish - Learning with Disagreement](https://detests-dis.github.io/).

## Table of Contents

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

- [Programs and folder structure](#programs-and-folder-structure)
  - [Main programs](#main-programs)
  - [Utils](#utils)
- [Datasets](#datasets)
  - [DETESTS](#detests)
  - [StereoHoax-ES corpus](#stereohoax-es-corpus)
- [Results](#results)
- [Reproduce](#reproduce)
  - [Setup](#setup)
  - [Split](#split)
  - [BERTs Fine-tuning](#berts-fine-tuning)
    - [Hyper-parameters](#hyper-parameters)
  - [Figures](#figures)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->


## Programs and folder structure


### Main programs

- `baselines.py`: Creates the [baselines](#baselines) for the tasks.
- `context_soft.py` Adds the context attributes and soft-labels to the datasets.
- `create_metrics.py`: Computes the metrics for the results files.
- `preprocess_split.py`: Preprocesses and splits the corpus as needed.

### Utils

- `config.py`: Contains feature names and constants.
- `data_pred.py`: Does the data processing and predictions for the fine-tuned models.
- `fine_tuning.py` Utils for the fine-tuning notebook.
- `io.py`: Parses inputs and outputs.
- `plots.py`: Plot utils.
- `results.py`: Contains a class to store the results and functions to compute
  metrics.
- `split.py`: Utils for split notebooks.
- `temperature_scaling.py`: Utils for [temperature scaling](https://github.com/gpleiss/temperature_scaling).
- `trainers.py`: Different HF Trainers and main train function.

- `scripts`: This folder contains bash scripts for [reproducibility](#reproduce).


The following notebooks are included:

- `notebooks/split_stereohoax.ipynb` splits the StereoHoax corpus (see
- `notebooks/split_detests.ipynb` splits the DETESTS corpus (see [split](#split)).
- `notebooks/analysis.ipynb` was the notebook used to create the tables
and figures for the paper, as well as to extract the relevant data to be qualitatively analyzed.
- `fine_tuning_hard_and_soft.ipynb`
  has the fine-tuning of the BERT models. It was originally run on free Google Colab with a T4 GPU.
  It was since ported to be used on a local machine.

## Datasets

If you are interested in the corpora **for research purposes** please [contact the authors](mailto:pol.pastells@ub.edu),
and we will provide the zip passwords.

### DETESTS

- `DETESTS_dataset_GS.xlsx` is the original unsplit data with `file_id` (news
  page where the comment came from) and only the majority voting labels. The
  `sentence_id` column contains both the comment id and position of the
  sentence. E.g., "13.02" is the second sentence in comment 13.

- `news_DETESTS.csv` maps `file_id` to the `news_title`.

- `train_with_disagreement_orig.csv` and `test_with_disagreement_orig.csv` are the
  original split data **with the 3 unaggregated annotations**. Here
  `sentence_id` is already split into `comment_id` and `sentence_pos`.
  These are the initial point for creating the further datasets.

  1. For some topics the majority vote didn't match the gold labels. During the
  annotation process weekly meeting were scheduled to discuss the annotation
  guidelines. Some gold labels were changed in theese meeting, without changing the
  individual annotators. This is fixed by randomly changing a single annotation so
  it matches the discussed gold label. See `detests-inconsistencies.py` notebook.

  2. `train.csv` and `test.csv` are created with `preprocess_split.py`

  3. When passed through `context_soft.py` the preprocessed files become
     `train_context_soft.csv` and `test_context_soft.csv`; and the unprocessed
     files become `train_with_disagreement_context_split.csv` and
     `test_with_disagreement_context_split.csv`.

  4. `train_with_disagreement_context_split.csv` is further [split](#split) in
     `val_split_context_soft.csv` and `train_split_context_soft.csv`.

### StereoHoax-ES corpus


- `stereoHoax-ES_goldstandard.csv` is the original data.
- `stereohoax_unaggregated.csv` has the 3 unaggregated annotations for
  "stereotype","implicit", "contextual" and "url".
- `train_val_split.csv`, `train_split.csv`, `val_split.csv` and `test_split.csv`
  are the [split](#split) sets, also with the unaggregated annotations.
  1. `context_soft.py` creates a `_context_soft` version for each one.
  2. `preprocess_split` takes `train_val_split_context_soft.csv` and
     `test_split_context_soft.csv` as inputs to create `train.csv` and
     `test.csv`.

## Results

Predictions for each model are stored in the `results` folder along with the
gold standard. Each model's results are separated into different CSV files,
with the predictions for each feature being named '<feature>\_pred'.

The overall metrics for all models are shown in the `results/metrics` folder.

These metrics can be recreated using the `create_metrics.py` script.

## Reproduce

### Setup

To set up the necessary environment and download required models, run
`scripts/setup.sh`.

### Split

For the baselines, the data is simply split into train and test sets. For the
BERT models, a validation set is also created.

For the StereoHoax corpus, the following splits are created: 70% train, 10%
validation, and 20% test. Different racial hoaxes are separated into different
sets to avoid data leakage and preserve the distribution of stereotypes. The
split is performed using `split_stereohoax.py`.

`split_stereohoax.py` works in the following way:

1. Finds combination of hoaxes that reach 70%, 20% and 10% of the data.
2. Finds which of these combinations has the most similar topic distribution to
   the original data.

The resulting splits are the following:

- Train_val - 80% = Train + val (used for baselines)
- Train - 70%: 'SP003', 'SP013', 'SP064', 'SP054', 'SP070', 'SP017', 'SP067',
  'SP043', 'SP036', 'SP048'
- Val - 10%: 'SP005', 'SP065', 'SP052', 'SP055', 'SP068'
- Test - 20%: 'SP057', 'SP015', 'SP049', 'SP047', 'SP010', 'SP014', 'SP009',
  'SP027', 'SP040', 'SP020', 'SP023', 'SP008', 'SP031'

The percentage of the whole dataset that each racial hoax (RH) contributes to
the splits is the following:

```python
Train = {
    'SP067': 0.93,
    'SP043': 27.97,
    'SP036': 8.32,
    'SP048': 0.06,
    'SP064': 14.73,
    'SP003': 16.69,
    'SP054': 0.02,
    'SP070': 0.04,
    'SP013': 0.02,
    'SP017': 0.07,
    'sum':  68.85,
}
Validation = {
    'SP052': 1.72,
    'SP068': 3.87,
    'SP005': 0.06,
    'SP065': 2.13,
    'SP055': 3.33,
    'sum':  11.10,
}
Test = {
    'SP010': 0.19,
    'SP008': 1.31,
    'SP014': 1.42,
    'SP027': 5.12,
    'SP015': 3.72,
    'SP009': 0.50,
    'SP040': 0.79,
    'SP031': 0.24,
    'SP020': 1.70,
    'SP023': 0.34,
    'SP047': 3.44,
    'SP049': 1.03,
    'SP057': 0.24,
    'sum':  20.04,
}
```

The DETESTS corpus is already split into 70% train and 30% test. The train set
is further split into train and validation (approximately 10%) while preserving
the proportion of stereotypes and not mixing news using `split_detests.py`. This
is done **after** running `context.py`, so the news_title is already added.

The percentage that each news contributes to the splits is:

```python
Train = {
    '20200708_MI': 0.14,
    '20170819_CR': 0.11,
    'ABC20210520': 0.09,
    '20200831_CR': 0.09,
    '20190716_CR': 0.08,
    '20200715b_CR': 0.05,
    'sum':         0.57,
}
Validation = {}
    '20200725_CR': 0.11,
    'sum':         0.11,
}
Test = {
    'EM20211106': 0.11,
    'EM20210519': 0.08,
    'ABC20211105': 0.08,
    'AD20200719': 0.03,
    'EC20210829': 0.03,
    'sum':        0.32,
}
```

### Baselines

`train.csv` and `test.csv` are used for the baselines. To obtain them run
`scripts/preprocess.sh`, which calls `preprocess_split.py`.

The following baselines are considered:

- All-zeros
- All-ones
- Weighted random classifier
- TFIDF (with only unigrams) + linear SVC (Support Vector Classifier)
- TFIDF (with n-grams with sizes 1 to 3) + linear SVC
- FastText vectorization + linear SVC

They can be run with the following commands:

```sh
python baselines.py -data stereohoax
python baselines.py -data detests
```


### BERTs Fine-tuning

To fine-tune the BERT models for both corpora, run the `fine_tuning_hard_and_soft.ipynb` notebook with the adequate inputs.

Although not mentioned in the paper, we used temperature scaling and you can see the implementation.
Likewise, both hard and soft labels are implemented.

#### Hyper-parameters

The models with hard labels uses a learning rate of 2e-5, while the model with
soft-labels uses 1e-5.

We keep the model with lowest loss.
