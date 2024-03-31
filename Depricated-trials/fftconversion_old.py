import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt


file_paths = []
current_dir = os.getcwd()
input_folder_path = os.path.join(current_dir, 'processed_data')

all_files = os.listdir(input_folder_path)
csv_files = list(filter(lambda f: f.startswith('withoutbaby_') and f.endswith('.csv'), all_files))
for i in csv_files:
    name = os.path.join(input_folder_path, i)
    file_paths.append(name)
print(file_paths)


# Initialize an empty list to store the DataFrames
dfs = []

# Loop through the file paths and append each DataFrame to the list
for file in file_paths:
    df = pd.read_csv(file, header=None, index_col=False)
    dfs.append(df)

# Concatenate all the DataFrames in the list
combined_df = pd.concat(dfs, ignore_index=True)

len(combined_df)
print(len(combined_df))

# Get the shape of the DataFrame
shape = combined_df.shape

# Number of rows
num_rows = shape[0]

# Number of columns
num_columns = shape[1]

# Print the number of rows and columns
print("Number of rows:", num_rows)
print("Number of columns:", num_columns)


# Select from the 17th column to the end
adc_data_selected_columns = combined_df.iloc[:, 16:].mean(axis=1)  # Python uses 0-based indexing


# Assuming `adc_data` is your pandas Series with ADC data
adc_array = adc_data_selected_columns.to_numpy()  # Convert the pandas Series to a numpy array

# Choose a window function - Hanning window in this case
window = np.hanning(len(adc_array))

# Apply the window function to your data
windowed_adc_data = adc_array * window

# Perform FFT on the windowed data
fft_result = np.fft.fft(windowed_adc_data)

# Frequency bins (assuming you know the sampling rate)
sampling_rate = 1000  # Example: 1000 Hz, replace with your actual sampling rate
n = len(adc_array)
freq = np.fft.fftfreq(n, d=1/sampling_rate)
# Calculate the magnitude and phase of the FFT result
magnitude = np.abs(fft_result)
phase = np.angle(fft_result)

# Create a DataFrame
fft_df = pd.DataFrame({
    'Frequency': freq,
    'FFT Magnitude': magnitude,
    'Phase': phase
})


# Add a new column to the fft_magnitude DataFrame for the binary label
# Set the value to 1 for presence of an infant with a carriage
fft_df['Infant_Presence'] = 0  # 1 for presence , and 0 for without baby case

len(fft_df)


plt.figure(figsize=(10, 6))  # Set the figure size for better readability
plt.plot(fft_df['Frequency'], fft_df['FFT Magnitude'])  # Plot positive frequency vs magnitude
plt.title('Magnitude Spectrum (Positive Frequencies)')  # Title of the plot
plt.xlabel('Frequency (Hz)')  # Label for the x-axis
plt.ylabel('Magnitude')  # Label for the y-axis
plt.grid(True)  # Show grid for better readability
plt.show()  # Display the plot

numpy_array = fft_df.to_numpy()
# Save the array to a file
np.save('withoutbaby_npy_array.npy', numpy_array)