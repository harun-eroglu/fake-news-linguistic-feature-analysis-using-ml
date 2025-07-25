import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

liar_df = pd.read_csv("liar_rf_feature_importance.csv")
isot_df = pd.read_csv("isot_rf_feature_importance.csv")

liar_df = liar_df.rename(columns={'importance': 'LIAR_RF'})
isot_df = isot_df.rename(columns={'importance': 'ISOT_RF'})

combined_df = pd.merge(liar_df, isot_df, on='feature')

combined_df.sort_values(by='ISOT_RF', ascending=False, inplace=True)

combined_df.to_csv("part3_feature_importance_combined.csv", index=False)

plt.figure(figsize=(8, max(6, len(combined_df) * 0.4)))
sns.heatmap(combined_df.set_index('feature'), annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={'label': 'Importance'})
plt.title("Feature Importance Heatmap – LIAR vs ISOT (Random Forest)")
plt.tight_layout()
plt.savefig("part3_feature_importance_combined_heatmap.png")
plt.close()

print("Combined feature importance heatmap saved as 'part3_feature_importance_combined_heatmap.png'")
