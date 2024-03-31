from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
# Sample ADC data (replace this with your actual ADC data)
adc_data = [0, 1, 0, 1, 1, 0, 0, 1, 1, 0]  # Example binary classification (0: baby not present, 1: baby present)
# Sample ground truth labels (replace this with your actual ground truth labels)
ground_truth = [0, 1, 0, 1, 1, 0, 1, 0, 1, 0]  # Example ground truth labels

# Constructing the confusion matrix
conf_matrix = confusion_matrix(ground_truth, adc_data)

# Print the confusion matrix
print("Confusion Matrix:")
print(conf_matrix)

# Create ConfusionMatrixDisplay object and plot the matrix
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
disp.plot();




y_true = [2, 0, 2, 2, 0, 1]
y_pred = [0, 0, 2, 2, 0, 2]
confusion_matrix1  =confusion_matrix(y_true, y_pred)
display = ConfusionMatrixDisplay(confusion_matrix1).plot()
plt.show()