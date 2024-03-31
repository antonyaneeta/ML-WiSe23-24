import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

# Load data for with baby
withbaby_data = np.load('withbaby_npy_array.npy')
withbaby_df = pd.DataFrame(withbaby_data, columns=['Frequency', 'FFT Magnitude', 'Phase', 'Infant_Presence'])

# Load data for without baby
withoutbaby_data = np.load('withoutbaby_npy_array.npy')
withoutbaby_df = pd.DataFrame(withoutbaby_data, columns=['Frequency', 'FFT Magnitude', 'Phase', 'Infant_Presence'])

# Combine both datasets
combined_df = pd.concat([withbaby_df, withoutbaby_df], ignore_index=True)

# Split features and target
X = combined_df.drop('Infant_Presence', axis=1)
y = combined_df['Infant_Presence']

# Splitting the dataset into the training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier
classifier.fit(X_train, y_train)

# Predictions
y_pred = classifier.predict(X_test)

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
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Display evaluation metrics
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1:.2f}")
