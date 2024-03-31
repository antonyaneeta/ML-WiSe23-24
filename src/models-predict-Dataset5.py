import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import roc_curve, precision_recall_curve, auc
from joblib import load

# Load data for with baby
withbaby_data = np.load('withshade_npy_array5.npy')
withbaby_df = pd.DataFrame(withbaby_data, columns=['Frequency', 'FFT Magnitude', 'Phase', 'Infant_Presence'])

# Load data for without baby
withoutbaby_data = np.load('withoutbaby_npy_array3.npy')
withoutbaby_df = pd.DataFrame(withoutbaby_data, columns=['Frequency', 'FFT Magnitude', 'Phase', 'Infant_Presence'])

print(len(withbaby_df))
print(len(withoutbaby_df))
# Combine both datasets
combined_df = pd.concat([withbaby_df, withoutbaby_df], ignore_index=True)

# Split features and target
X = combined_df.drop('Infant_Presence', axis=1)
y = combined_df['Infant_Presence']
# Splitting the dataset into the training set and test set
print(len(X))
print(len(y))
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=42)

# Load the saved MLPClassifier model
Loaded_mlp = load('mlpClassifier_model_trained1.pkl')

# Evaluate the MLPClassifier model on the entire dataset X
y_pred_mlp = Loaded_mlp.predict(X)
accuracy_mlp = accuracy_score(y, y_pred_mlp)
precision_mlp = precision_score(y, y_pred_mlp)
recall_mlp = recall_score(y, y_pred_mlp)
f1_mlp = f1_score(y, y_pred_mlp)
tn_mlp, fp_mlp, fn_mlp, tp_mlp = confusion_matrix(y, y_pred_mlp).ravel()
FNR_mlp = fn_mlp / (fn_mlp + tp_mlp)
FPR_mlp = fp_mlp / (fp_mlp + tn_mlp)
cv_scores_mlp = cross_val_score(Loaded_mlp, X, y, cv=5)

# Print evaluation metrics for MLPClassifier
print("MLPClassifier Evaluation:")
print("False Negative Rate:", FNR_mlp)
print("False Positive Rate:", FPR_mlp)
print("Cross-validation scores:", cv_scores_mlp)
print("Mean CV score:", cv_scores_mlp.mean())

# Calculate confusion matrix for MLPClassifier
conf_matrix_mlp = confusion_matrix(y, y_pred_mlp)

# Plot and save confusion matrix for MLPClassifier
plt.figure(figsize=(6, 6))
sns.heatmap(conf_matrix_mlp, annot=True, fmt='d', cmap='Greens_r', cbar=False,
            xticklabels=['No Babydoll', 'With Babydoll'], yticklabels=['No Babydoll', 'With Babydoll'])
plt.title('Confusion Matrix - MLPClassifier (Dataset 5)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('mlp_dataset5_confusion_matrix_DS5.png')
plt.show()


# Display evaluation metrics for MLPClassifier
print(f"Accuracy: {accuracy_mlp:.2f}")
print(f"Precision: {precision_mlp:.2f}")
print(f"Recall: {recall_mlp:.2f}")
print(f"F1-score: {f1_mlp:.2f}")



# Load the saved RandomForestClassifier model
Loaded_rf = load('random_forest_model_trained1.pkl')

# Evaluate the RandomForestClassifier model on the entire dataset X
y_pred_rf = Loaded_rf.predict(X)
accuracy_rf = accuracy_score(y, y_pred_rf)
precision_rf = precision_score(y, y_pred_rf)
recall_rf = recall_score(y, y_pred_rf)
f1_rf = f1_score(y, y_pred_rf)
tn_rf, fp_rf, fn_rf, tp_rf = confusion_matrix(y, y_pred_rf).ravel()
FNR_rf = fn_rf / (fn_rf + tp_rf)
FPR_rf = fp_rf / (fp_rf + tn_rf)
cv_scores_rf = cross_val_score(Loaded_rf, X, y, cv=5)

# Print evaluation metrics for RandomForestClassifier
print("\nRandomForestClassifier Evaluation:")
print("False Negative Rate:", FNR_rf)
print("False Positive Rate:", FPR_rf)
print("Cross-validation scores:", cv_scores_rf)
print("Mean CV score:", cv_scores_rf.mean())

# Calculate confusion matrix for RandomForestClassifier
conf_matrix_rf = confusion_matrix(y, y_pred_rf)


# Plot and save confusion matrix for RandomForestClassifier
plt.figure(figsize=(6, 6))
sns.heatmap(conf_matrix_rf, annot=True, fmt='d', cmap='Blues_r', cbar=False,
            xticklabels=['No Babydoll', 'With Babydoll'], yticklabels=['No Babydoll', 'With Babydoll'])
plt.title('Confusion Matrix - RandomForestClassifier (Dataset 5)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('random_forest_dataset5_confusion_matrix_DS5.png')
plt.show()

# Display evaluation metrics for RandomForestClassifier
print(f"Accuracy: {accuracy_rf:.2f}")
print(f"Precision: {precision_rf:.2f}")
print(f"Recall: {recall_rf:.2f}")
print(f"F1-score: {f1_rf:.2f}")

# Compute ROC curve and ROC area for MLPClassifier
fpr_mlp, tpr_mlp, _ = roc_curve(y, y_pred_mlp)
roc_auc_mlp = auc(fpr_mlp, tpr_mlp)

# Compute Precision-Recall curve and area under the curve for MLPClassifier
precision_mlp, recall_mlp, _ = precision_recall_curve(y, y_pred_mlp)
pr_auc_mlp = auc(recall_mlp, precision_mlp)

# Compute ROC curve and ROC area for RandomForestClassifier
fpr_rf, tpr_rf, _ = roc_curve(y, y_pred_rf)
roc_auc_rf = auc(fpr_rf, tpr_rf)

# Compute Precision-Recall curve and area under the curve for RandomForestClassifier
precision_rf, recall_rf, _ = precision_recall_curve(y, y_pred_rf)
pr_auc_rf = auc(recall_rf, precision_rf)

# Plot ROC curve for both classifiers
plt.figure()
plt.plot(fpr_mlp, tpr_mlp, color='darkorange', lw=2, label='ROC curve - MLP (area = %0.2f)' % roc_auc_mlp)
plt.plot(fpr_rf, tpr_rf, color='green', lw=2, label='ROC curve - RF (area = %0.2f)' % roc_auc_rf)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - MLP vs RF (Dataset 5)')
plt.legend(loc="lower right")
plt.savefig('mlp_vs_rf_roc_curve_DS5.png')
plt.show()

# Plot Precision-Recall curve for both classifiers
plt.figure()
plt.plot(recall_mlp, precision_mlp, color='blue', lw=2, label='Precision-Recall curve - MLP (area = %0.2f)' % pr_auc_mlp)
plt.plot(recall_rf, precision_rf, color='red', lw=2, label='Precision-Recall curve - RF (area = %0.2f)' % pr_auc_rf)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall Curve - MLP vs RF (Dataset 5)')
plt.legend(loc="lower left")
plt.savefig('mlp_vs_rf_precision_recall_curve_DS5.png')
plt.show()

# Count the occurrences of each label in the test set
label_counts = y.value_counts()

# Count the occurrences of each predicted label
predicted_label_counts = pd.Series(y_pred_mlp).value_counts()


# Count the occurrences of each label in the test set for MLP
label_counts_mlp = y.value_counts()

# Count the occurrences of each predicted label for MLP
predicted_label_counts_mlp = pd.Series(y_pred_mlp).value_counts()

# Count the occurrences of each predicted label for RF
predicted_label_counts_rf = pd.Series(y_pred_rf).value_counts()

# Create a figure and axis object
fig, ax = plt.subplots(figsize=(12, 6))

# Set bar width
bar_width = 0.3

# Plot bar chart for MLP
ax.bar(label_counts_mlp.index - bar_width/2, label_counts_mlp.values, width=bar_width, alpha=0.5, label='Actual Labels - MLP')
ax.bar(predicted_label_counts_mlp.index + bar_width/2, predicted_label_counts_mlp.values, width=bar_width, alpha=0.5, label='Predicted Labels - MLP')

# Plot bar chart for RF
ax.bar(label_counts_mlp.index - bar_width/2, label_counts_mlp.values, width=bar_width, alpha=0.5, label='Actual Labels - RF')
ax.bar(predicted_label_counts_rf.index + bar_width/2, predicted_label_counts_rf.values, width=bar_width, alpha=0.5, label='Predicted Labels - RF')

# Set labels and title
plt.xlabel('Label')
plt.ylabel('Count')
plt.title('Distribution of Actual and Predicted Labels for MLP and RF Classifiers (Dataset 5)')
plt.xticks(label_counts_mlp.index, ['No Babydoll', 'With Babydoll'])
plt.legend()

# Show plot
plt.tight_layout()
plt.savefig('label_distribution_bar_graph_combined_DS5.png')
plt.show()