# Apply Feature Extraction to LIAR and ISOT datasets

import pandas as pd
from ii_feature_extraction_module import FakeNewsFeatureExtractor, extract_features_from_dataframe

liar_path = "i_liar_binarized.csv"
isot_path = "i_isot_downsampled.csv"

liar_df = pd.read_csv(liar_path, encoding="utf-8")
isot_df = pd.read_csv(isot_path, encoding="utf-8")

print(f"LIAR dataset shape: {liar_df.shape}")
print(f"ISOT dataset shape: {isot_df.shape}")

extractor = FakeNewsFeatureExtractor()

print("Extracting features for LIAR dataset...")
liar_features_df = extract_features_from_dataframe(liar_df, text_column='text')
liar_features_df['label'] = liar_df['label'].values

liar_features_df.to_csv("ii_liar_features.csv", index=False, encoding="utf-8")
print("LIAR feature matrix saved as 'ii_liar_features.csv'")

print("Extracting features for ISOT dataset...")
isot_features_df = extract_features_from_dataframe(isot_df, text_column='text')
isot_features_df['label'] = isot_df['label'].values

isot_features_df.to_csv("ii_isot_features.csv", index=False, encoding="utf-8")
print("ISOT feature matrix saved as 'ii_isot_features.csv'")
