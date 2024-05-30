#!/bin/bash


# -- Train and test for baselines --

## DETESTS
data=detests
python3 preprocess_split.py -data $data -datafile train_with_disagreement.csv
mv data/$data/clean.csv data/$data/train.csv
python3 preprocess_split.py -data $data -datafile test_with_disagreement.csv
mv data/$data/clean.csv data/$data/test.csv

# Add contexts to DETESTS after preprocess_split and to Stereohoax before
# to avoid re-tokenizing several times the same sentences
python3 context_soft.py

## Stereohoax
data=stereohoax
python3 preprocess_split.py -data $data -datafile train_val_split_context_soft.csv
mv data/$data/clean.csv data/$data/train.csv
python3 preprocess_split.py -data $data -datafile test_split_context_soft.csv
mv data/$data/clean.csv data/$data/test.csv
