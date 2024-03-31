import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
# Replace 'your_array_file.npy' with the actual file path of your saved NumPy array
file_path = 'withbaby_npy_array.npy'

# Load the NumPy array from the file
loaded_array = np.load(file_path, mmap_mode='r')

# Now 'loaded_array' contains the NumPy array data that was saved in the file
dataframe_withbaby = pd.DataFrame(loaded_array,columns=['Frequency','FFT Magnitude','Phase','Infant_Presence'])

print(len(dataframe_withbaby))

# Replace 'your_array_file.npy' with the actual file path of your saved NumPy array
file_path = 'withoutbaby_npy_array.npy'

# Load the NumPy array from the file
loaded_array_1 = np.load(file_path, mmap_mode='r')
# Now 'loaded_array' contains the NumPy array data that was saved in the file
dataframe_withoutbaby = pd.DataFrame(loaded_array_1,columns=['Frequency','FFT Magnitude','Phase','Infant_Presence'])
print(len(dataframe_withoutbaby))



# True Positives (TP) and True Negatives (TN) are determined by the counts of the datasets
true_positives = len(dataframe_withbaby)
true_negatives = len(dataframe_withoutbaby)

# False Positives (FP) and False Negatives (FN) are assumed to be 0 in this scenario
false_positives = 0
false_negatives = 0

# Printing the confusion matrix in a more formatted way
print("Confusion Matrix:")
print()
print(f"{'':<15}{'Predicted NO':<15}{'Predicted YES':<15}")
print(f"{'Actual NO':<15}{true_negatives:<15}{false_positives:<15}")
print(f"{'Actual YES':<15}{false_negatives:<15}{true_positives:<15}")


# Assuming you have already calculated true_positives, true_negatives,
# false_positives, and false_negatives as before

# Constructing the confusion matrix as a 2D array
conf_matrix = [
    [true_negatives, false_positives],
    [false_negatives, true_positives]
]

# Labels for the axes
labels = ['0', '1'] # 1 baby present in seat , 0 -no baby doll in carriage

# Creating a heatmap using seaborn
plt.figure(figsize=(6, 6))  # Setting the figure size
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues_r', xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()



# Clubbing both datasets
dataset_1 = pd.concat([dataframe_withbaby,dataframe_withoutbaby])
dataset_1 = dataset_1.reset_index(drop=True)

print(len(dataset_1))

# Prepare the data by normalizing it and splitting it into training and testing sets
X = dataset_1.iloc[:, :-1]  # Input features
y = dataset_1.iloc[:, -1]  # Target labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# Train Random Forest Classifier model
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train, epochs=10, batch_size=512)

# Evaluate Random Forest Classifier model
y_pred = classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1:.2f}")


cm = confusion_matrix(y_test, y_pred)

# Creating a heatmap using seaborn
plt.figure(figsize=(6, 6))  # Setting the figure size
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues_r', xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()

print(cm)
