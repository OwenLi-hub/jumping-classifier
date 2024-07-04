import h5py
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d

# set path to hdf5_data.h5 to get the train and test dataset from
hdf5_path = 'hdf5_data.h5'

# Define function to apply moving average filter
def apply_MA(data, window_size=25):
    filtered_data = uniform_filter1d(data, size=window_size, axis=0, mode='nearest')
    return filtered_data

# Define function to extract features
def extract_features(data):
    # Extracting the required features for all columns in the data
    features = {}
    Combined_Data = data.flatten()

    # extract that max and min value
    # range, mean, median, variance, skewness, and standered deivation
    features['max'] = Combined_Data.max()
    features['min'] = Combined_Data.min()
    features['range'] = Combined_Data.ptp()  
    features['mean'] = Combined_Data.mean()
    features['median'] = np.median(Combined_Data)
    features['variance'] = Combined_Data.var()
    features['skewness'] = pd.Series(Combined_Data).skew()
    features['std'] = Combined_Data.std()

    # extract percentile of 25, 50 and 75 percent
    for p in [25, 50, 75]:
        features[f'percentile_{p}'] = np.percentile(Combined_Data, p)

    return features

# Normalized the features using Z-score standardization
def normalize_features(features):
    # Create an empty dictionary to store the processed data
    normalize_datas = {}
    # Iterate over each key feature and place it in dictionary 'features'
    for key, value in features.items():
        # Check if the key contains 'mean' or 'std' in its feature
        # We don't want to normalize std or mean because that would undo the Z-score standardization
        # directly place it in dictionary if detected
        if 'mean' in key or 'std' in key: 
            normalize_datas[key] = value
        else:
            # Extract the mean and standard deviation for Z-score standardization calculation
            feature_name = key.split('_')[0]  # Extract the feature name (e.g., 'max', 'min', etc.)
            mean_key = f'mean_{feature_name}'
            std_key = f'std_{feature_name}'
            mean = features.get(mean_key, 0)  # Use .get() method to avoid KeyError if key doesn't exist
            std = features.get(std_key, 1)     # Use .get() method to avoid KeyError if key doesn't exist, and default to 1 to prevent division by zero
            normalize_datas[key] = (value - mean) / std

    return normalize_datas



# Define a function to read data from the HDF5 dataset
def read_hdf5(hdf5_path, dataset_path):
    with h5py.File(hdf5_path, 'r') as hdf:
        data = hdf[dataset_path][:]
    return data

# Segment the data length into 5 seconds for the csv files with segment_length 500
def segment_data(data, segment_length=500):
    number_segments = len(data) // segment_length
    segments = [data[i*segment_length:(i+1)*segment_length] for i in range(number_segments)]
    return segments

# Load the training and testing data from HDF5
train_data = read_hdf5(hdf5_path, '/dataset/train/train_data')
test_data = read_hdf5(hdf5_path, '/dataset/test/test_data')

# Process the data (apply moving average filter)
Processed_TrainData = apply_MA(train_data, window_size=25)
Processed_TestData = apply_MA(test_data, window_size=25)

# Segment the processed data
train_segments = segment_data(Processed_TrainData)
test_segments = segment_data(Processed_TestData)

# Extract features and normalize them for training data
train_features = [extract_features(segment) for segment in train_segments]
train_normalized_features = [normalize_features(features) for features in train_features]
train_DataFrame = pd.DataFrame(train_normalized_features)
train_DataFrame.to_csv('train_features.csv', index=False)

# Extract features and normalize them for testing data
test_features = [extract_features(segment) for segment in test_segments]
test_normalized_features = [normalize_features(features) for features in test_features]
test_DataFrame = pd.DataFrame(test_normalized_features)
test_DataFrame.to_csv('test_features.csv', index=False)