import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Load data from CSV files
excel_owen = 'Owen_Data.csv'
data_frame1 = pd.read_csv(excel_owen, usecols=[4])

excel_mathhew = 'Matthew_Data.csv'
data_frame2 = pd.read_csv(excel_mathhew, usecols=[4])

excel_thomas = 'Thomas_Data.csv'
data_frame3 = pd.read_csv(excel_thomas, usecols=[4])

# Define segment length
segment_length = 500

# Segment the data frame by 5-second intervals and combine them
segments1 = [data_frame1[i:i+segment_length] for i in range(0, len(data_frame1), segment_length)]
segments2 = [data_frame2[i:i+segment_length] for i in range(0, len(data_frame2), segment_length)]
segments3 = [data_frame3[i:i+segment_length] for i in range(0, len(data_frame3), segment_length)]

# Shuffle individual segments within each segment list
for i in range(len(segments1)):
    segments1[i] = segments1[i].sample(frac=1, random_state=None)

for i in range(len(segments2)):
    segments2[i] = segments2[i].sample(frac=1, random_state=None)

for i in range(len(segments3)):
    segments3[i] = segments3[i].sample(frac=1, random_state=None)

# Concatenate segments to form the complete dataset
all_segments = pd.concat(segments1 + segments2 + segments3)

# Convert segments to a NumPy array without padding
shuffle_segments = all_segments.sample(frac=1, random_state=None)

# Split the segments into 90% training and 10% testing 
train_segments, test_segments = train_test_split(shuffle_segments, test_size=0.1, random_state=False)


# Create HDF5 file and store data
with h5py.File('hdf5_data.h5', 'w') as hdf:

    #group of DataSet for Owen
    Owen_group = hdf.create_group ('/Owen')
    Owen_group.create_dataset('Owens_dataset', data=data_frame1)

    #group ofDataSet for Matthew
    Matthew_group = hdf.create_group ('/Matthew')
    Matthew_group.create_dataset('Matthews_dataset', data=data_frame2)

    #group of DataSet for Thomas
    Thoma_group = hdf.create_group ('/Thomas')
    Thoma_group.create_dataset('Thomas_dataset', data=data_frame3)


    # Creat an group called datase that stores both the train and store dataset
    dataset_group = hdf.create_group('dataset')

    train_group = dataset_group.create_group('train')
    train_group.create_dataset('train_data', data=train_segments)

    test_group = dataset_group.create_group('test')
    test_group.create_dataset('test_data', data=test_segments)





# Path to the HDF5 file
#hdf5_path = 'hdf5_data.h5'

# Open the HDF5 file in read mode
#with h5py.File(hdf5_path, 'r') as hdf:
    #train_data = hdf['/dataset/train/train_data'][:]
    
    #test_data = hdf['/dataset/test/test_data'][:]

#train_df = pd.DataFrame(train_data.reshape(train_data.shape[0], -1))
#train_df.to_csv('train_data.csv', index=False)

#test_df = pd.DataFrame(test_data.reshape(test_data.shape[0], -1))
#test_df.to_csv('test_data.csv', index=False)
