#MAIN
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import joblib

# Load dataset
df = pd.read_csv("expanded_tep_based_fault_data.csv") 
X = df.drop(columns=['Fault_Label'])  
y = df['Fault_Label']  

# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Apply SMOTE to balance minority faults
minority_classes = np.unique(y_train)[1:]  
smote_strategy = {c: max(np.bincount(y_train)) for c in minority_classes}  
smote = SMOTE(sampling_strategy=smote_strategy, random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Apply controlled undersampling to Class 0 (remove 50%, not 70%)
undersample = RandomUnderSampler(sampling_strategy={0: int(np.bincount(y_train_resampled)[0] * 0.5)}, random_state=42)
X_train_resampled, y_train_resampled = undersample.fit_resample(X_train_resampled, y_train_resampled)

# Hyperparameter tuning with RandomizedSearchCV
param_grid = {
    'n_estimators': [300, 400],  
    'max_depth': [20, 30],  
    'min_samples_split': [2, 5],  
    'min_samples_leaf': [1, 2],  
    'bootstrap': [True, False]
}

random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=param_grid,
    n_iter=10,
    cv=3,
    verbose=1,
    n_jobs=-1
)

random_search.fit(X_train_resampled, y_train_resampled)  
best_params = random_search.best_params_

# Adjust class weights to prevent excessive misclassification of Class 0
class_weights = {0:2, 1:3, 2:2, 3:2.5, 4:2.5, 5:1}

# Train final model with adjusted class weights
rf_model = RandomForestClassifier(**best_params, class_weight=class_weights, random_state=42)
rf_model.fit(X_train_resampled, y_train_resampled)

# K-Fold Cross Validation (5 folds)
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []
for train_idx, val_idx in kf.split(X_train_resampled, y_train_resampled):
    X_train_fold, X_val_fold = X_train_resampled.iloc[train_idx], X_train_resampled.iloc[val_idx]
    y_train_fold, y_val_fold = y_train_resampled.iloc[train_idx], y_train_resampled.iloc[val_idx]
    
    rf_model.fit(X_train_fold, y_train_fold)
    y_val_pred = rf_model.predict(X_val_fold)
    score = accuracy_score(y_val_fold, y_val_pred)
    cv_scores.append(score)

# Display cross-validation results
print(f"Cross-Validation Accuracy Scores: {cv_scores}")
print(f"Mean Accuracy: {np.mean(cv_scores):.4f}")

# Predict probabilities for test set
y_probs = rf_model.predict_proba(X_test)

# Adjusted decision thresholds to reduce false positives
thresholds = {0: 0.7, 1: 0.2, 2: 0.3, 3: 0.25, 4: 0.3, 5: 0.5}
y_pred_adjusted = np.array([
    np.argmax([y_probs[i, j] if y_probs[i, j] >= thresholds[j] else 0 for j in range(y_probs.shape[1])])
    for i in range(y_probs.shape[0])
])

# Evaluate final model
print("Updated Classification Report:")
print(classification_report(y_test, y_pred_adjusted, zero_division=1))

print("Updated Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_adjusted))

# Fault Detection Rate Calculation
conf_matrix = confusion_matrix(y_test, y_pred_adjusted)
true_negatives = conf_matrix[0, 0]
false_negatives = sum(conf_matrix[:, 0]) - true_negatives
total_faults = sum(sum(conf_matrix[1:]))  

detection_rate = (total_faults - false_negatives) / total_faults * 100
print(f"Updated Fault Detection Rate: {detection_rate:.2f}%")

# Downtime Reduction Calculation
downtime_per_missed_fault = 2  
historical_downtime = false_negatives * downtime_per_missed_fault  
new_downtime = (false_negatives * 0.85) * downtime_per_missed_fault  
downtime_reduction = ((historical_downtime - new_downtime) / historical_downtime) * 100

print(f"Updated Estimated Downtime Reduction: {downtime_reduction:.2f}%")

# Save trained model
joblib.dump(rf_model, "random_forest_fault_detection_final_v3.pkl")

print("Final Model Trained, Validated, and Saved.")
