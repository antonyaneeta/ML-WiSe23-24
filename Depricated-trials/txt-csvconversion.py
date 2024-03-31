import csv
import os
import numpy as np
import pandas as pd
#
import os
import csv
import glob

# Get the current directory
current_dir = os.getcwd()

# Define the folder paths
input_folder_path = os.path.join(current_dir, 'primary_data')
output_folder_path = os.path.join(current_dir, 'processed_data')

# Ensure the output folder exists, create if it doesn't
os.makedirs(output_folder_path, exist_ok=True)

#
# Find all files with names like "withbaby_*.txt" in the input folder
txt_files = glob.glob(os.path.join(input_folder_path, 'withoutbaby_*.txt'))

for txt_file_path in txt_files:
    # Get the base filename without extension
    base_filename = os.path.splitext(os.path.basename(txt_file_path))[0]

    # Form the output CSV file path
    csv_file_path = os.path.join(output_folder_path, base_filename + '.csv')

    # Open the text file for reading
    with open(txt_file_path, 'r') as txt_file:
        # Read the lines from the text file
        lines = txt_file.readlines()

    # Open the CSV file for writing
    with open(csv_file_path, 'w', newline='') as csv_file:
        # Create a CSV writer object
        csv_writer = csv.writer(csv_file)

        # Write each line from the text file to the CSV file
        for line in lines:
            # Strip any leading/trailing whitespace and split the line by space
            # to get the individual values
            values = line.strip().split()
            # Write the values to the CSV file
            csv_writer.writerow(values)

print("Conversion complete.")

# current_dir = os.getcwd()
# folder_path = os.path.join(current_dir, 'primary_data')
# file_path = os.path.join(folder_path, 'withbaby.txt')
# # Open the text file for reading
# with open(file_path, 'r') as txt_file:
#     # Read the lines from the text file
#     lines = txt_file.readlines()
#
# # Open the CSV file for writing
# with open('withbaby.csv', 'w', newline='') as csv_file:
#     # Create a CSV writer object
#     csv_writer = csv.writer(csv_file)
#
#     # Write each line from the text file to the CSV file
#     for line in lines:
#         # Strip any leading/trailing whitespace and split the line by comma
#         # to get the individual values
#         values = [value.strip() for value in line.split(' ')]
#         # Write the values to the CSV file
#         csv_writer.writerow(values)

print("Conversion completed successfully!")
# dfs= []
# df = pd.read_csv("withbaby.csv", header=None, index_col=False)
# dfs.append(df)
# print(dfs)


