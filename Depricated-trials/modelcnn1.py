import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV

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
param_grid = {
    'n_estimators': [50, 100, 150],  # Number of trees in the forest
    'max_depth': [None, 10, 20],  # Maximum depth of the tree
    'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4]  # Minimum number of samples required to be at a leaf node
}
# Initialize RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
# Perform grid search with cross-validation
grid_search = GridSearchCV(estimator=classifier, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
# Get the best parameters and best estimator
best_params = grid_search.best_params_
best_estimator = grid_search.best_estimator_
# Evaluate the best model on the test set
y_pred = best_estimator.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

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
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('mlp_trained_confusion_matrix_plot.png')
plt.show()


# Display evaluation metrics
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1:.2f}")


print("Best parameters found by grid search:")
print(best_params)
print(f"Best accuracy on validation set: {grid_search.best_score_:.2f}")
print(f"Accuracy on test set: {accuracy:.2f}")

# Save the best_estimator to a file
joblib.dump(best_estimator, 'random_forest_model1.pkl')