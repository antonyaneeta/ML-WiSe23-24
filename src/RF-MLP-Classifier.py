import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Load data for with baby
withbaby_data = np.load('withbaby_npy_array.npy')
withbaby_df = pd.DataFrame(withbaby_data, columns=['Frequency', 'FFT Magnitude', 'Phase', 'Infant_Presence'])
num_rows_withbaby=len(withbaby_df)
# Initialize an empty DataFrame with the desired number of rows
withbaby_measurements_df = pd.DataFrame(index=range(num_rows_withbaby))

# # Dictionary of new measurements with their base values
# new_measurements = {
#     'SensorToBabyHead': 43
# }
#
# # Randomly generate values within ±3 range for each measurement
# for column, base_value in new_measurements.items():
#     # Use numpy to generate a random number in the range [base_value - 3, base_value + 3) for each row
#     random_values = np.random.uniform(base_value - 1, base_value + 1, size=num_rows_withbaby)
#
#     # Assign these values to the corresponding column in the DataFrame
#     withbaby_measurements_df[column] = random_values
#
# # Concatenate the DataFrames side by side
# withbaby_df = pd.concat([withbaby_df, withbaby_measurements_df], axis=1)

#Load data for without baby
withoutbaby_data = np.load('withoutbaby_npy_array.npy')
withoutbaby_df = pd.DataFrame(withoutbaby_data, columns=['Frequency', 'FFT Magnitude', 'Phase', 'Infant_Presence'])
num_rows_withoutbaby=len(withoutbaby_df)
# # Initialize an empty DataFrame with the desired number of rows
# withoutbaby_measurements_df = pd.DataFrame(index=range(num_rows_withoutbaby))
#
# # Dictionary of new measurements with their base values
# new_measurements = {
#
# }
#
# # Randomly generate values within ±3 range for each measurement
# for column, base_value in new_measurements.items():
#     # Use numpy to generate a random number in the range [base_value - 3, base_value + 3) for each row
#     random_values = np.random.uniform(base_value - 1, base_value + 1, size=num_rows_withoutbaby)
#
#     # Assign these values to the corresponding column in the DataFrame
#     withoutbaby_measurements_df[column] = random_values
#
# withoutbaby_measurements_df['SensorToBabyHead'] = 0
# # Concatenate the DataFrames side by side
# withoutbaby_df = pd.concat([withoutbaby_df, withoutbaby_measurements_df], axis=1)

# Combine both datasets
combined_df = pd.concat([withbaby_df, withoutbaby_df], ignore_index=True)


# Split features and target
X = combined_df.drop('Infant_Presence', axis=1)
y = combined_df['Infant_Presence']
# Splitting the dataset into the training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
param_grid = {
    'n_estimators': [10, 50, 100],  # Number of trees in the forest
    'max_depth': [None, 5, 10],  # Maximum depth of the tree
    'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
    'min_samples_leaf': [2, 4, 6]  # Minimum number of samples required to be at a leaf node
}
# Initialize RandomForestClassifier
classifier = RandomForestClassifier(random_state=42)
# Perform grid search with cross-validation
grid_search = GridSearchCV(estimator=classifier, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Get the best parameters and best estimator
best_params = grid_search.best_params_
best_estimator = grid_search.best_estimator_
# Evaluate the best model on the test set
y_pred = best_estimator.predict(X_test)
#accuracy = accuracy_score(y_test, y_pred)

# Train the classifier
# classifier.fit(X_train, y_train)

# Predictions
# y_pred = classifier.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plotting the confusion matrix
plt.figure(figsize=(6, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues_r', cbar=False,
            xticklabels=['No Baby', 'With Baby'], yticklabels=['No Baby', 'With Baby'])
plt.title('RFClassifer Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('rfclassifier_trained_confusion_matrix_plot1.png')
plt.show()


# Display evaluation metrics
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1:.2f}")

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

FNR = fn / (fn + tp)
FPR = fp / (fp + tn)

print("False Negative Rate:", FNR)
print("False Positive Rate:", FPR)

print("Best parameters found by grid search:")
print(best_params)
print(f"Best accuracy on validation set: {grid_search.best_score_:.2f}")
print(f"Accuracy on test set: {accuracy:.2f}")

# Save the best_estimator to a file
joblib.dump(best_estimator, 'random_forest_model_trained1.pkl')


# Define the preprocessing pipeline (optional but recommended)
pipeline = make_pipeline(StandardScaler())  # Standardize features

# Preprocess the training data
X_train_preprocessed = pipeline.fit_transform(X_train)

# Preprocess the testing data (using the same pipeline)
X_test_preprocessed = pipeline.transform(X_test)

#Define the MLP classifier
# mlp_classifier = MLPClassifier(hidden_layer_sizes=(4,), activation='relu', solver='adam', alpha=0.001,
#                                 batch_size='auto', learning_rate='constant', learning_rate_init=0.001,
#                                 max_iter=200, shuffle=True, random_state=42, tol=1e-4, verbose=False,
#                                 early_stopping=False, validation_fraction=0.1)

mlp_classifier = MLPClassifier(hidden_layer_sizes=(7,), activation='logistic', solver='adam', alpha=0.01,
                                batch_size='auto', learning_rate='constant', learning_rate_init=0.01,
                                max_iter=100, shuffle=True, random_state=42, tol=1e-3, verbose=False,
                                early_stopping=False, validation_fraction=0.1)

# Fit the classifier to your data
mlp_classifier.fit(X_train, y_train)

# Predict on test data
y_pred1 = mlp_classifier.predict(X_test)
# Calculate evaluation metrics
accuracy1 = accuracy_score(y_test, y_pred1)
precision1 = precision_score(y_test, y_pred1)
recall1 = recall_score(y_test, y_pred1)
f1_1 = f1_score(y_test, y_pred1)
# Display evaluation metrics
print(f"Accuracy: {accuracy1:.2f}")
print(f"Precision: {precision1:.2f}")
print(f"Recall: {recall1:.2f}")
print(f"F1-score: {f1_1:.2f}")
tn1, fp1, fn1, tp1 = confusion_matrix(y_test, y_pred1).ravel()

FNR1 = fn1 / (fn1 + tp1)
FPR1 = fp1 / (fp1 + tn1)

print("False Negative Rate:", FNR1)
print("False Positive Rate:", FPR1)
# Calculate confusion matrix
conf_matrix_1 = confusion_matrix(y_test, y_pred1)

# Plotting the confusion matrix
plt.figure(figsize=(6, 6))
sns.heatmap(conf_matrix_1, annot=True, fmt='d', cmap='Blues_r', cbar=False,
            xticklabels=['No Baby', 'With Baby'], yticklabels=['No Baby', 'With Baby'])
plt.title('MLPClassiifer Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('mlp_trained_confusion_matrix_plot_MLPClassifier1.png')
plt.show()

# Save the mpl classifier to a file
joblib.dump(mlp_classifier, 'mlpClassifier_model_trained1.pkl')
# 1. Performance Metrics Table
performance_table = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-score'],
    'Random Forest': [accuracy, precision, recall, f1],
    'MLP Classifier': [accuracy1, precision1, recall1, f1_1]
})
print(performance_table)

# 2. ROC Curve
from sklearn.metrics import roc_curve, auc
plt.figure(figsize=(8, 6))
fpr, tpr, _ = roc_curve(y_test, best_estimator.predict_proba(X_test)[:, 1])
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='darkorange', lw=2, label='Random Forest (AUC = %0.2f)' % roc_auc)

fpr1, tpr1, _ = roc_curve(y_test, mlp_classifier.predict_proba(X_test)[:, 1])
roc_auc1 = auc(fpr1, tpr1)
plt.plot(fpr1, tpr1, color='blue', lw=2, label='MLP Classifier (AUC = %0.2f)' % roc_auc1)

plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.savefig('ROC Curve.png')
plt.show()

# 3. Precision-Recall Curve
from sklearn.metrics import precision_recall_curve
precision, recall, _ = precision_recall_curve(y_test, best_estimator.predict_proba(X_test)[:, 1])
plt.plot(recall, precision, color='darkorange', lw=2, label='Random Forest')

precision1, recall1, _ = precision_recall_curve(y_test, mlp_classifier.predict_proba(X_test)[:, 1])
plt.plot(recall1, precision1, color='blue', lw=2, label='MLP Classifier')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.savefig('Precision-Recall Curve.png')
plt.show()
