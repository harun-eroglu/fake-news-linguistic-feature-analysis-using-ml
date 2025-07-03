# Fake News Detection using Linguistic Feature Analysis with SVM and RF

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

## How to Run

Run scripts sequentially in the following order:
- i_liar_binarized.py
- i_isot_downsampled.py
- ii_feature_extraction_module.py (imported by next script)
- ii_feature_extraction_execution.py
- ii_heatmap.py
- part3_training_pipeline.py

## Datasets

Due to file size limitations on GitHub, please download the original, unprocessed datasets required to run the preprocessing and feature extraction scripts from the following Google Drive link:

[Download LIAR and ISOT datasets (Google Drive)](https://drive.google.com/drive/folders/1DOMW3gtJIwVQnDlF0IrPB5ZE01MAl6wq?usp=sharing)

**Instructions:**
1. Download the ZIP file and extract it.
2. Place the extracted folders (`liar` and `isot`) inside your project directory.

## Result files

Additionally, preprocessed datasets and all final result CSV files (feature matrices, classification reports for SVM and RF on both datasets, feature importance outputs, best hyperparameter selection, paired t-test, cross-domain generalizability results) can be downloaded from the following link:

[Download preprocessed data and results (Google Drive)](https://drive.google.com/drive/folders/1larNKuQV_IR1qi49BQem0Aawl-D-xGXm?usp=sharing)

**Instructions:**
- Place the preprocessed data files into appropriate directories if you want to skip preprocessing scripts.
- The results folder includes outputs from `part3_training_pipeline.py` for direct inspection without rerunning models.

## Author

- Harun Eroglu
