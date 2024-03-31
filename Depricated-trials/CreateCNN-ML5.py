import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import regularizers
from keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv1D, Dropout, MaxPooling1D, Flatten, Dense
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV

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
# Initialize an empty DataFrame with the desired number of rows
withoutbaby_measurements_df = pd.DataFrame(index=range(num_rows_withoutbaby))

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Assuming X_train and X_test are DataFrames
# Reshape input data
#X_train = X_train.values.reshape(32, len(X_train), 3)  # Reshape to (batch_size, sequence_length, num_features)
#X_test = X_test.values.reshape(32, 3, 1)
#
# # Create a CNN model
# modelCNN = tf.keras.Sequential([
#     tf.keras.layers.Conv1D(32, 3, activation='relu', input_shape=(32, 0,)),  # Convolutional layer with 32 filters and kernel size 3
#     tf.keras.layers.MaxPooling1D(2),  # Max pooling layer
#     tf.keras.layers.Conv1D(64, 3, activation='relu'),  # Another convolutional layer with 64 filters and kernel size 3
#     tf.keras.layers.MaxPooling1D(2),  # Max pooling layer
#     tf.keras.layers.Conv1D(64, 3, activation='relu'),  # Another convolutional layer with 64 filters and kernel size 3
#     tf.keras.layers.Flatten(),  # Flatten layer to prepare for fully connected layers
#     tf.keras.layers.Dense(64, activation='relu'),  # Fully connected layer with 64 units
#     tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer with sigmoid activation for binary classification
# ])
#
# # Compile the model
# modelCNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# Reshape input data
# Reshape input data
X_train_reshaped = X_train.values.reshape(X_train.shape[0], X_train.shape[1],1 )
X_test_reshaped = X_test.values.reshape(X_test.shape[0], X_train.shape[1],1 )
# # One-hot encode target data
# y_train_encoded = to_categorical(y_train, 2)
# y_test_encoded = to_categorical(y_test,0)

verbose, epochs, batch_size = 1, 10, 32
modelCNN = Sequential()

# CNN Model architecture
modelCNN.add(Conv1D(filters=64, kernel_size=3, activation="relu", input_shape=(3, 1)))
modelCNN.add(MaxPooling1D(pool_size=1))

modelCNN.add(Flatten())
modelCNN.add(Dense(1052, activation='relu'))
modelCNN.add(Dense(1, activation='sigmoid'))  # Single output neuron for binary classification

# Compile the model
modelCNN.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
modelCNN.fit(X_train_reshaped, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)

# Evaluate model
loss,accuracy = modelCNN.evaluate(X_test, y_test, batch_size=batch_size, verbose=verbose)
accuracy = accuracy* 100.0  # Use accuracy metric, which is the second element in the evaluation result tuple
print('Accuracy of Model: ', accuracy)
# Print model summary
modelCNN.summary()
# Fit the classifier to your data with adjusted parameters
#modelCNN.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=0.1)

# Predict on test data
y_pred1 = modelCNN.predict(X_test)


# Convert probabilities to binary class labels
y_preds_binary = np.round(y_pred1)
y_preds_binary=y_preds_binary.flatten()
# Evaluate the model on the test data

#loss, accuracy = modelCNN.evaluate(X_test, y_test, verbose=0)

# Print the results
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

# Calculate evaluation metrics
accuracy1 = accuracy_score(y_test, y_preds_binary)
precision1 = precision_score(y_test, y_preds_binary)
recall1 = recall_score(y_test, y_preds_binary)
f1_1 = f1_score(y_test, y_preds_binary)
# Display evaluation metrics
print(f"Accuracy: {accuracy1:.2f}")
print(f"Precision: {precision1:.2f}")
print(f"Recall: {recall1:.2f}")
print(f"F1-score: {f1_1:.2f}")

# Calculate confusion matrix
conf_matrix_cnn = confusion_matrix(y_test, y_preds_binary)

# Plotting the confusion matrix
plt.figure(figsize=(6, 6))
sns.heatmap(conf_matrix_cnn, annot=True, fmt='d', cmap='Blues_r', cbar=False,
            xticklabels=['No Baby', 'With Baby'], yticklabels=['No Baby', 'With Baby'])
plt.title('CNN Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('cnn_trained_DS1_confusion_matrix_plot-v1.png')
plt.show()

# Save the cnn model classifier to a file
modelCNN.save('modelCNN_trained1.h5')