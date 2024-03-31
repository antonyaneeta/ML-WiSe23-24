#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 14:10:29 2023

@author: kirankumarathirala
"""

import numpy as np
import os
import csv
import glob
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

# Get the current working directory
current_dir = os.getcwd()


# Function to read FFT data from a text file
def read_fft_data(filename):
    fft_data = []
    with open(filename, 'r') as file:
        for line in file:
            # Split the line by comma and convert each value to float
            fft_values = [float(value) for value in line.strip().split(',')]

            # Append each value to fft_values along with the label 0 or 1
            if "adc_no" in filename:
                fft_values.append(0)
            else:
                fft_values.append(1)

            # Append fft_values as a separate array to fft_data
            fft_data.append(fft_values)

    return np.array(fft_data)


# Load the FFT data from multiple text files and combine them into a single data set
folder_path = os.path.join(current_dir, 'Data')
data = []
# Filename pattern for the FFT data text files
file_pattern = os.path.join(folder_path, 'adc_*.txt')
# Read FFT data from the file
for filename in glob.glob(file_pattern):
    n = read_fft_data(filename)
    data.append(n)

data = np.array(data)

# Prepare the data by normalizing it and splitting it into training and testing sets
X = data[:, :-1]  # Input features
y = data[:, -1]  # Target labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# Train Random Forest Classifier model
clasiffier = RandomForestClassifier(n_estimators=100, random_state=42)
clasiffier.fit(X_train, y_train)

# Evaluate Random Forest Classifier model
y_pred = clasiffier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1:.2f}")

# Calculate and print confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Calculate TP, FP, FN, TN, TPR, TNR, FPR, FNR
TP = cm[1][1]
FP = cm[0][1]
FN = cm[1][0]
TN = cm[0][0]

TPR = TP / (TP + FN)
TNR = TN / (TN + FP)
FPR = FP / (FP + TN)
FNR = FN / (FN + TP)
ACC = (TP + TN) / (TP + FP + FN + TN)
print(f"True Positive (TP): {TP:.2f}")
print(f"True Negative (TN): {TN:.2f}")
print(f"False Positive (FP): {FP:.2f}")
print(f"False Negative (FN): {FN:.2f}")
print(f"True Positive Rate (TPR): {TPR:.2f}")
print(f"True Negative Rate (TNR): {TNR:.2f}")
print(f"False Positive Rate (FPR): {FPR:.2f}")
print(f"False Negative Rate (FNR): {FNR:.2f}")
print(f"Accuracy (ACC): {ACC:.2f}")

# Create ConfusionMatrixDisplay object and plot the matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot();

# Save trained model to file
model_output = os.path.join(current_dir, 'output/random_forest_model.joblib')
joblib.dump(clasiffier, model_output)

file_test = os.path.join(folder_path, 'adc_20000_fft.txt')
# Load test data from file

test_data = read_fft_data(file_test)

# Load trained model from file
clasiffier = joblib.load(model_output)
predict_data = test_data[:, :-1]
# Make predictions on test data
y_prediction = clasiffier.predict(predict_data)

print("predicted labels list:", y_prediction)

output_filepath = os.path.join(current_dir, 'output/predicted_output.csv')
output_prediction_img = os.path.join(current_dir, 'output/Prediction_output.png')
# Open the CSV file for writing
with open(output_filepath, 'w', newline='') as file:
    # Define the fieldnames
    fieldnames = ['prediction label', 'prediction']

    # Create a DictWriter object
    writer = csv.DictWriter(file, fieldnames=fieldnames)

    # Write the headers to the CSV file
    writer.writeheader()

    # Write the data to the CSV file
    for value in y_prediction:
        if (value == 1):
            writer.writerow({'prediction label': value, 'prediction': "Baby in the Seat"})
        elif (value == 0):
            writer.writerow({'prediction label': value, 'prediction': "Empty Seat"})
# Print predicted labels
y_prediction_str = []
for predicted_value in y_prediction:
    if (predicted_value == 1):
        y_prediction_str.append("Baby in the seat")
    elif (predicted_value == 0):
        y_prediction_str.append("Empty seat")

# Create a new figure for the pie chart
plt.figure()

# Count the number of "baby" and "empty" values
num_baby = y_prediction_str.count("Baby in the seat")
num_empty = y_prediction_str.count("Empty seat")

# Create a bar chart of the data
labels = ["Baby in the seat", "Empty seat"]
values = [num_baby, num_empty]
bar_width = 0.1
plt.bar(labels, values, width=bar_width)

# Add text labels to the top of each bar
plt.text(labels[0], values[0], values[0], ha='center', va='bottom', fontsize=10)
plt.text(labels[1], values[1], values[1], ha='center', va='bottom', fontsize=10)
# Add axis labels and a title
plt.xlabel("Prediction")
plt.ylabel("Prediction value")
plt.title("Prediction Results")

# save the plot
plt.savefig(output_prediction_img)
# Show the plot
plt.show()
