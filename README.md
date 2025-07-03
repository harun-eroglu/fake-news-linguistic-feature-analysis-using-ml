# Fake News Detection using Linguistic Feature Analysis

This repository contains all scripts used for a fake news detection project comparing traditional machine learning models (SVM, Random Forest) trained on handcrafted linguistic features across LIAR and ISOT datasets. The study also tests cross-domain generalizability to evaluate feature robustness under realistic deployment conditions.

## Repository Files

### i_liar_binarized.py
Preprocesses the LIAR dataset by merging train/valid/test splits, binarizing labels into fake (1) vs real (0), and saving a cleaned CSV containing only text and label columns.

### i_isot_downsampled.py
Loads ISOT True/Fake CSV files, assigns binary labels, downsamples each class to balance the dataset, removes only the parenthesized source tags, and saves the cleaned dataset as CSV.

### ii_feature_extraction_module.py
Defines the `FakeNewsFeatureExtractor` class with POS-based, structural, and emotional feature extraction methods, using NLTK and NRC Emotion Lexicon.

### ii_feature_extraction_execution.py
Applies the feature extraction pipeline to the preprocessed LIAR and ISOT datasets, generating CSV files containing feature matrices with text and labels.

### heatmap.py
Combines Random Forest feature importance outputs from LIAR and ISOT into a single dataframe and visualizes their comparison as a heatmap using Seaborn.

### part3_training_pipeline.py
Implements the final training pipeline: data loading, standardization, SVM and Random Forest training with GridSearchCV, cross-domain generalizability tests, statistical significance tests, and saves all results as CSVs for further analysis and reporting.

## Overview

This study investigates the effectiveness of traditional machine learning models trained on interpretable handcrafted linguistic features for fake news detection. Three feature categories were used:

- POS-based stylistic features
- Structural features
- Emotional features (NRC Emotion Lexicon)

The project further evaluates whether models trained on one domain (political claims or full news articles) generalize to another domain without fine-tuning.

## How to Run

Run scripts sequentially in the following order:
- i_liar_binarized.py
- i_isot_downsampled.py
- ii_feature_extraction_module.py (imported by next script)
- ii_feature_extraction_execution.py
- heatmap.py
- part3_training_pipeline.py

## Datasets

- **LIAR dataset:** Political claims labeled for truthfulness.
- **ISOT dataset:** Full news articles labeled as fake or real.

Ensure datasets are placed in the correct paths as defined in each script.

## Results

The `part3_training_pipeline.py` script outputs include:
- Classification reports for SVM and RF on both datasets
- Best hyperparameter selections
- Feature importance rankings
- Cross-domain generalizability results
- Statistical significance (paired t-test) results

## Repository Name Suggestion

`fake-news-linguistic-feature-analysis`

This name concisely reflects the project’s scope and methodology.

---

Maintained by [Your Name]. Contact for further clarifications or collaboration proposals.

---

This version is now in **GitHub Markdown format**, ready for direct copy-paste to your repository README.md file. Let me know if you want badge templates or license section next.

