# LIAR Binarize
# Cleaned: keep only 'text' + 'label'

import pandas as pd

train_path = "/mnt/c/Users/Harun Eroğlu/Desktop/Machine Learning for NLP/final_report/liar/train.tsv"
valid_path = "/mnt/c/Users/Harun Eroğlu/Desktop/Machine Learning for NLP/final_report/liar/valid.tsv"
test_path = "/mnt/c/Users/Harun Eroğlu/Desktop/Machine Learning for NLP/final_report/liar/test.tsv"

columns = ['id','label','statement','subject','speaker','speaker_job_title','state_info','party_affiliation',
           'barely_true_counts','false_counts','half_true_counts','mostly_true_counts','pants_on_fire_counts','context']

train_df = pd.read_csv(train_path, sep='\t', names=columns, encoding="utf-8")
valid_df = pd.read_csv(valid_path, sep='\t', names=columns, encoding="utf-8")
test_df = pd.read_csv(test_path, sep='\t', names=columns, encoding="utf-8")

liar_df = pd.concat([train_df, valid_df, test_df], ignore_index=True)

def label_mapper(x):
    if x in ['pants-fire', 'false', 'barely-true']:
        return 1
    else:
        return 0

liar_df['binary_label'] = liar_df['label'].apply(label_mapper)

liar_df_cleaned = liar_df[['statement','binary_label']].rename(columns={'statement':'text','binary_label':'label'})

liar_df_cleaned = liar_df_cleaned.sort_values(by='label').reset_index(drop=True)

print(liar_df_cleaned['label'].value_counts())

liar_df_cleaned.to_csv("i_liar_binarized.csv", index=False, encoding="utf-8")

print("LIAR dataset binarized, sorted, and saved as 'i_liar_binarized.csv'")
