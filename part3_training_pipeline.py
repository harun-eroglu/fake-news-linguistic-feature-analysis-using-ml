
# 1. Import Libraries

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from scipy.stats import ttest_rel, ttest_ind, pearsonr, spearmanr
import joblib
import time

start_time = time.time()


# 2. Load Datasets

liar_df = pd.read_csv("ii_liar_features.csv", encoding="utf-8")
isot_df = pd.read_csv("ii_isot_features.csv", encoding="utf-8")

def prepare_data(df):
    X = df.drop(columns=['text', 'label'])
    y = df['label']
    return X, y

X_liar, y_liar = prepare_data(liar_df)
X_isot, y_isot = prepare_data(isot_df)


# 3. Train-Test Split

X_liar_train, X_liar_test, y_liar_train, y_liar_test = train_test_split(
    X_liar, y_liar, test_size=0.2, random_state=42, stratify=y_liar)

X_isot_train, X_isot_test, y_isot_train, y_isot_test = train_test_split(
    X_isot, y_isot, test_size=0.2, random_state=42, stratify=y_isot)


# 4. Normalization

scaler_liar = StandardScaler()
X_liar_train_scaled = scaler_liar.fit_transform(X_liar_train)
X_liar_test_scaled = scaler_liar.transform(X_liar_test)
joblib.dump(scaler_liar, "scaler_liar.pkl")

scaler_isot = StandardScaler()
X_isot_train_scaled = scaler_isot.fit_transform(X_isot_train)
X_isot_test_scaled = scaler_isot.transform(X_isot_test)
joblib.dump(scaler_isot, "scaler_isot.pkl")


# 5. Model Training – SVM

svm_params = [
    {'kernel': ['linear'], 'C': [0.01, 0.1, 1, 10]},
    {'kernel': ['rbf'], 'C': [0.01, 0.1, 1, 10], 'gamma': ['scale', 0.001, 0.01, 0.1, 1]}
]

try:
    print("Starting SVM training on LIAR dataset...")
    svm_grid_liar = GridSearchCV(SVC(), svm_params, cv=10, scoring='f1_macro', n_jobs=-1)
    svm_grid_liar.fit(X_liar_train_scaled, y_liar_train)
    print("SVM LIAR training completed!")
except Exception as e:
    print(f"SVM training (LIAR) failed: {e}")

try:
    print("Starting SVM training on ISOT dataset...")
    svm_grid_isot = GridSearchCV(SVC(), svm_params, cv=10, scoring='f1_macro', n_jobs=-1)
    svm_grid_isot.fit(X_isot_train_scaled, y_isot_train)
    print("SVM ISOT training completed!")
except Exception as e:
    print(f"SVM training (ISOT) failed: {e}")

# best params only to optimize memory
svm_best_df = pd.DataFrame([
    {'dataset': 'liar', 'model': 'svm', 'best_params': svm_grid_liar.best_params_, 'best_score': svm_grid_liar.best_score_},
    {'dataset': 'isot', 'model': 'svm', 'best_params': svm_grid_isot.best_params_, 'best_score': svm_grid_isot.best_score_}
])
svm_best_df.to_csv("svm_best_params.csv", index=False)


# 6. Model Training – Random Forest

rf_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

try:
    print("Starting Random Forest training on LIAR dataset...")
    rf_grid_liar = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=10, scoring='f1_macro', n_jobs=-1)
    rf_grid_liar.fit(X_liar_train_scaled, y_liar_train)
    print("Random Forest LIAR training completed!")
except Exception as e:
    print(f"RF training (LIAR) failed: {e}")

try:
    print("Starting Random Forest training on ISOT dataset...")
    rf_grid_isot = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=10, scoring='f1_macro', n_jobs=-1)
    rf_grid_isot.fit(X_isot_train_scaled, y_isot_train)
    print("Random Forest ISOT training completed!")
except Exception as e:
    print(f"RF training (ISOT) failed: {e}")

rf_best_df = pd.DataFrame([
    {'dataset': 'liar', 'model': 'rf', 'best_params': rf_grid_liar.best_params_, 'best_score': rf_grid_liar.best_score_},
    {'dataset': 'isot', 'model': 'rf', 'best_params': rf_grid_isot.best_params_, 'best_score': rf_grid_isot.best_score_}
])
rf_best_df.to_csv("rf_best_params.csv", index=False)


# 7. Evaluation on Test Set

def evaluate_model(model, X_test, y_test, dataset_name, model_name):
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    df_report.to_csv(f"{dataset_name}_{model_name}_classification_report.csv")
    return report['accuracy'], report['macro avg']['f1-score'], report['macro avg']['precision'], report['macro avg']['recall']

results = []
acc, f1, prec, rec = evaluate_model(svm_grid_liar.best_estimator_, X_liar_test_scaled, y_liar_test, "liar", "svm")
results.append(['liar', 'svm', acc, f1, prec, rec])

acc, f1, prec, rec = evaluate_model(rf_grid_liar.best_estimator_, X_liar_test_scaled, y_liar_test, "liar", "rf")
results.append(['liar', 'rf', acc, f1, prec, rec])

acc, f1, prec, rec = evaluate_model(svm_grid_isot.best_estimator_, X_isot_test_scaled, y_isot_test, "isot", "svm")
results.append(['isot', 'svm', acc, f1, prec, rec])

acc, f1, prec, rec = evaluate_model(rf_grid_isot.best_estimator_, X_isot_test_scaled, y_isot_test, "isot", "rf")
results.append(['isot', 'rf', acc, f1, prec, rec])

results_df = pd.DataFrame(results, columns=['dataset', 'model', 'accuracy', 'f1_macro', 'precision_macro', 'recall_macro'])
results_df.to_csv("part3_test_results.csv", index=False)


# 8. Paired t-test (SVM vs RF)

svm_scores_liar = cross_val_score(svm_grid_liar.best_estimator_, X_liar_train_scaled, y_liar_train, cv=10, scoring='f1_macro')
rf_scores_liar = cross_val_score(rf_grid_liar.best_estimator_, X_liar_train_scaled, y_liar_train, cv=10, scoring='f1_macro')
ttest_liar = ttest_rel(svm_scores_liar, rf_scores_liar)

svm_scores_isot = cross_val_score(svm_grid_isot.best_estimator_, X_isot_train_scaled, y_isot_train, cv=10, scoring='f1_macro')
rf_scores_isot = cross_val_score(rf_grid_isot.best_estimator_, X_isot_train_scaled, y_isot_train, cv=10, scoring='f1_macro')
ttest_isot = ttest_rel(svm_scores_isot, rf_scores_isot)

ttest_df = pd.DataFrame({
    'dataset': ['liar', 'isot'],
    'p_value': [ttest_liar.pvalue, ttest_isot.pvalue],
    'statistic': [ttest_liar.statistic, ttest_isot.statistic]
})
ttest_df.to_csv("part3_paired_ttest_results.csv", index=False)


# 9. Feature Importance (RF)

def save_feature_importance(model, X_train, filename):
    importances = model.feature_importances_
    feature_names = X_train.columns
    fi_df = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values(by='importance', ascending=False)
    fi_df.to_csv(filename, index=False)

save_feature_importance(rf_grid_liar.best_estimator_, X_liar_train, "liar_rf_feature_importance.csv")
save_feature_importance(rf_grid_isot.best_estimator_, X_isot_train, "isot_rf_feature_importance.csv")


# 10. Cross-Domain Generalizability Analysis

print("Starting Cross-Domain Generalizability Analysis...")

def cross_domain_test(model_source, X_target, y_target, scaler_source, scaler_target, source_name, target_name, model_name):
    """Test model trained on source domain on target domain"""
    X_target_scaled_by_source = scaler_source.transform(X_target)
    y_pred = model_source.predict(X_target_scaled_by_source)
    report = classification_report(y_target, y_pred, output_dict=True)
    
    return {
        'source_domain': source_name,
        'target_domain': target_name,
        'model': model_name,
        'accuracy': report['accuracy'],
        'f1_macro': report['macro avg']['f1-score'],
        'precision_macro': report['macro avg']['precision'], 
        'recall_macro': report['macro avg']['recall']
    }

# Cross-domain tests
cross_domain_results = []

# LIAR -> ISOT
cross_domain_results.append(cross_domain_test(
    svm_grid_liar.best_estimator_, X_isot_test, y_isot_test, 
    scaler_liar, scaler_isot, 'liar', 'isot', 'svm'
))

cross_domain_results.append(cross_domain_test(
    rf_grid_liar.best_estimator_, X_isot_test, y_isot_test,
    scaler_liar, scaler_isot, 'liar', 'isot', 'rf'
))

# ISOT -> LIAR  
cross_domain_results.append(cross_domain_test(
    svm_grid_isot.best_estimator_, X_liar_test, y_liar_test,
    scaler_isot, scaler_liar, 'isot', 'liar', 'svm'
))

cross_domain_results.append(cross_domain_test(
    rf_grid_isot.best_estimator_, X_liar_test, y_liar_test,
    scaler_isot, scaler_liar, 'isot', 'liar', 'rf'
))

cross_domain_df = pd.DataFrame(cross_domain_results)
cross_domain_df.to_csv("part3_cross_domain_results.csv", index=False)

print("Cross-Domain vs Same-Domain Performance Comparison:")
for result in cross_domain_results:
    same_domain = results_df[
        (results_df['dataset'] == result['target_domain']) & 
        (results_df['model'] == result['model'])
    ]
    if not same_domain.empty:
        same_f1 = same_domain.iloc[0]['f1_macro']
        cross_f1 = result['f1_macro']
        performance_drop = same_f1 - cross_f1
        
        print(f"{result['model'].upper()}: {result['source_domain']}→{result['target_domain']}")
        print(f"  Same-domain F1: {same_f1:.3f}")
        print(f"  Cross-domain F1: {cross_f1:.3f}")
        print(f"  Performance drop: {performance_drop:.3f}")
        print()

print("Cross-Domain Generalizability Analysis Completed.")

total_minutes = (time.time() - start_time) / 60
print(f"Total execution time: {total_minutes:.1f} minutes")