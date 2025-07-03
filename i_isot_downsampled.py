# ISOT Downsampling and Binarize
# Cleaned: keep only 'text' + 'label'

import pandas as pd
import os
import re

true_path = "isot/True.csv"
fake_path = "isot/Fake.csv"

true_df = pd.read_csv(true_path, encoding="utf-8")
fake_df = pd.read_csv(fake_path, encoding="utf-8")

true_df['label'] = 0
fake_df['label'] = 1

isot_df = pd.concat([true_df, fake_df], ignore_index=True)

print("Original class distribution:")
print(isot_df['label'].value_counts())

def remove_source(text):
    return re.sub(r'\([^)]+\)\s*-\s*', '', text)

isot_df['text'] = isot_df['text'].apply(remove_source)

isot_df_downsampled = isot_df.groupby('label', group_keys=False).apply(lambda x: x.sample(6000, random_state=42))
isot_df_downsampled = isot_df_downsampled[['text', 'label']]

print("\nDownsampled class distribution:")
print(isot_df_downsampled['label'].value_counts())

isot_df_downsampled.to_csv("i_isot_downsampled.csv", index=False, encoding="utf-8")

print("\nDownsampled, cleaned, and source-removed ISOT dataset saved as 'i_isot_downsampled.csv'")
