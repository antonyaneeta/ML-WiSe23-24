import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from keras.models import load_model

from matplotlib.colors import LinearSegmentedColormap

# Load data for with baby
withbaby_data = np.load('withbaby_npy_array3.npy')
withbaby_df = pd.DataFrame(withbaby_data, columns=['Frequency', 'FFT Magnitude', 'Phase', 'Infant_Presence'])

# Load data for without baby
withoutbaby_data = np.load('withoutbaby_npy_array3.npy')
withoutbaby_df = pd.DataFrame(withoutbaby_data, columns=['Frequency', 'FFT Magnitude', 'Phase', 'Infant_Presence'])

# Combine both datasets
combined_df = pd.concat([withbaby_df, withoutbaby_df], ignore_index=True)

# Split features and target
X = combined_df.drop('Infant_Presence', axis=1)
y = combined_df['Infant_Presence']
# Splitting the dataset into the training set and test set
print(len(X))
print(len(y))
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Load the saved cnn model
Loaded_CNN = load_model('modelCNN_trained1.h5')

# Evaluate the best model on the test set
y_pred_dataset3 = Loaded_CNN.predict(X)

# Convert probabilities to binary class labels
y_pred_dataset3_binary = np.round(y_pred_dataset3)
y_pred_dataset3_binary=y_pred_dataset3_binary.flatten()
accuracy = accuracy_score(y, y_pred_dataset3_binary)

# Calculate evaluation metrics
accuracy = accuracy_score(y, y_pred_dataset3_binary)
precision = precision_score(y, y_pred_dataset3_binary)
recall = recall_score(y, y_pred_dataset3_binary)
f1 = f1_score(y, y_pred_dataset3_binary)

# Calculate confusion matrix
conf_matrix = confusion_matrix(y, y_pred_dataset3_binary)

# Plotting the confusion matrix
# Create pastel colormap
# Create pastel red colormap
colors = ["#FFCCCC", "#FF0000", "#FF9999", "#FF6666", "#FF3333"]  # Example pastel colors
cmap_pastel = LinearSegmentedColormap.from_list("pastel", colors, N=256)

plt.figure(figsize=(6, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap=cmap_pastel, cbar=False,
            xticklabels=['No Babydoll', 'With Babydoll'], yticklabels=['No Babydoll', 'With Babydoll'])
plt.title('Confusion Matrix-from pretrained CNNModel-dataset3')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('cnnmodel_dataset3_confusion_matrix_-final-optimized1.png')
plt.show()


# Display evaluation metrics
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1:.2f}")

tn, fp, fn, tp = confusion_matrix(y, y_pred_dataset3_binary).ravel()

FNR = fn / (fn + tp)
FPR = fp / (fp + tn)

print("False Negative Rate:", FNR)
print("False Positive Rate:", FPR)

# Compute ROC curve and ROC area
fpr, tpr, _ = roc_curve(y, y_pred_dataset3.flatten())
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for CNN')
plt.legend(loc="lower right")
plt.savefig('roc_curve_CNN_DS3.png')  # Save the ROC curve plot
plt.show()

# Compute Precision-Recall curve and area under the curve
precision, recall, _ = precision_recall_curve(y, y_pred_dataset3.flatten())
pr_auc = auc(recall, precision)

# Plot Precision-Recall curve
plt.figure()
plt.plot(recall, precision, color='blue', lw=2, label='Precision-Recall curve (area = %0.2f)' % pr_auc)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall Curve for CNN')
plt.legend(loc="lower left")
plt.savefig('precision_recall_curve_CNN_DS3.png')  # Save the Precision-Recall curve plot
plt.show()


