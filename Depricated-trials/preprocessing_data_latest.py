#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 18:15:13 2023

@author: kirankumarathirala
"""


import os

current_dir = os.getcwd()
folder_path = os.path.join(current_dir, 'primary_data')
folder_path_modified = os.path.join(current_dir, 'Data')

for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    if filename.endswith(".txt"):
        with open(file_path, "r") as infile:
            lines = infile.readlines()[1:]  # skip first line
            file_new = os.path.join(folder_path_modified, filename)
            with open(file_new, "w") as outfile:
                for line in lines:
                    fields = line.strip().split("\t")
                    if "withbaby" in filename:
                        fields.append("2")
                    elif "withoutbaby" in filename:
                        fields.append("1")

                    # join the fields together into a single string
                    line_str = "\t".join(fields)

                    # Removing Version string from the data
                    numeric_values = line_str.split()
                    if 'V0.2' in numeric_values:
                        numeric_values.remove('V0.2')
                    elif 'V0,2' in numeric_values:
                        numeric_values.remove('V0,2')
                    numeric_values = [(value.replace(',', '.')) for value in numeric_values]


                    # join the numeric values together into a single string
                    output = "\t".join(numeric_values)

                    # write the output string to the output file
                    outfile.write(output + "\n")
