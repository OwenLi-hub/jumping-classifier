import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the data from the CSV file
df = pd.read_csv("Owen_Data.csv")


# the code below contains the plot of raw data from the sensor CSV file
# Plot for Linear Acceleration Z (subplot 1)
plt.subplot(2, 2, 1)
plt.plot(df['Time (s)'], df['Linear Acceleration z (m/s^2)'], label='Z-Acceleration', color='darkgoldenrod')
plt.xlabel('Time (s)')
plt.ylabel('Acceleration (m/s^2)')
plt.title('Z-Acceleration vs. Time')
plt.legend()
plt.grid(True)

# Plot for Linear Acceleration Y (subplot 2)
plt.subplot(2, 2, 2)
plt.plot(df['Time (s)'], df['Linear Acceleration y (m/s^2)'], label='Y-Acceleration', color='g')
plt.xlabel('Time (s)')
plt.ylabel('Acceleration (m/s^2)')
plt.title('Y-Acceleration vs. Time')
plt.legend()
plt.grid(True)

# Plot for Linear Acceleration Y (subplot 2)
plt.subplot(2, 2, 3)
plt.plot(df['Time (s)'], df['Linear Acceleration x (m/s^2)'], label='X-Acceleration', color='navy')
plt.xlabel('Time (s)')
plt.ylabel('Acceleration (m/s^2)')
plt.title('Y-Acceleration vs. Time')
plt.legend()
plt.grid(True)

# Plot for Absolute Acceleration (subplot 3)
plt.subplot(2, 2, 4)
plt.plot(df['Time (s)'], df['Absolute acceleration (m/s^2)'], label='Absolute Acceleration', color='r')
plt.xlabel('Time (s)')
plt.ylabel('Acceleration (m/s^2)')
plt.title('Absolute Acceleration vs. Time')
plt.legend()
plt.grid(True)


# Show the plot
plt.tight_layout()
plt.show()











# the plot that shows all 10 extracted values as an ploted bar graph
dataset_name = "train_features.csv"
dataset = pd.read_csv(dataset_name)

data = dataset.iloc[:, :-1]
labels = dataset.iloc[:,-1]

fig, ax = plt.subplots(ncols=4,nrows=4, figsize=(20,10))

for i in range(13):
    data.hist(ax=ax.flatten()[i])

fig.tight_layout()
plt.show()













# the plot that seprate walking and jumping based on acceleration of 5m/s^2
# Create a mask for values under 5 m/s^2
mask_under_5 = df['Absolute acceleration (m/s^2)'] < 5

# Plot values under 5 m/s^2 in blue
plt.plot(df.loc[mask_under_5, 'Time (s)'], df.loc[mask_under_5, 'Absolute acceleration (m/s^2)'],
         label='Absolute Acceleration for walking', color='blue')

# Plot values over 5 m/s^2 in red
plt.plot(df.loc[~mask_under_5, 'Time (s)'], df.loc[~mask_under_5, 'Absolute acceleration (m/s^2)'],
         label='Absolute Acceleration for jumping', color='red')
plt.xlabel('Time (s)')
plt.ylabel('Acceleration (m/s^2)')
plt.title('Walking and Jumping reprsentation')
plt.legend()
plt.grid(True)
plt.show()
















# MA graph of filtering noises
# Load the noisy signal data
dataset = pd.read_csv("Owen_Data.csv")
noisy_signal = dataset.iloc[:, 4].values

# Window sizes
window_sizes = [5, 31, 51]

# Apply average filter for window size 5
filtered_signal_5 = pd.Series(noisy_signal).rolling(window=5).mean()

# Apply average filter for window size 31
filtered_signal_31 = pd.Series(noisy_signal).rolling(window=25).mean()

# Apply average filter for window size 51
filtered_signal_51 = pd.Series(noisy_signal).rolling(window=40).mean()

# Plot the graph
t = np.arange(len(noisy_signal))
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(t, noisy_signal, label='Original Noisy Signal', alpha=0.5)
ax.plot(t, filtered_signal_5, label='Moving Average (Window Size 5)')
ax.plot(t, filtered_signal_31, label='Moving Average (Window Size 25)')
ax.plot(t, filtered_signal_51, label='Moving Average (Window Size 50)')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Z-Accleration (m/s^2)')
ax.set_title('Original Noisy Signal vs. Moving Average Filtered Signals')
ax.legend()
ax.grid(True)
plt.show()