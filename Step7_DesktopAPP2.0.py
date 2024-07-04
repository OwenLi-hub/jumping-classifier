import tkinter as tk
from tkinter import filedialog
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.ndimage import uniform_filter1d
from sklearn.linear_model import LogisticRegression



# set up the overall GUI applicaion
# This section would be the front end of the GUi where styling and placment of button will be p;aced
# graph frames and csv frame will also be made in here
class SensorDataLabeler:
    def __init__(self, master):
        self.master = master
        self.master.title("Sensor Data Labeler") # set title
        self.master.configure(bg='#add8e6') # set colour
        self.file_path = None

        # Frame for file selection
        self.top_frame = tk.Frame(master)
        self.top_frame.pack()

        # Creaction if the file select button
        self.select_button = tk.Button(self.top_frame, text="Select CSV File", command=self.select_file, bg="yellow", fg="blue")
        self.select_button.grid(row=0, column=0, padx=5, pady=5)
        self.file_label = tk.Label(self.top_frame, text="No file selected")
        self.file_label.grid(row=0, column=1, padx=5, pady=5)

        # creat frame for the displayed labeled csv file and plot
        self.bottom_frame = tk.Frame(master)
        self.bottom_frame.pack()

        # creat overall Frame for plot
        self.plot_frame = tk.Frame(self.bottom_frame)
        self.plot_frame.pack(side=tk.LEFT)

        # Create and intliazed the empty plot
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.ax.set_title('Sensor Data Plot with Predicted Labels')
        self.ax.set_xlabel('Time')
        self.ax.set_ylabel('Value')
        self.line, = self.ax.plot([], [], label='Sensor Data')
        self.ax.legend()
        
        # Create a canvas for ploting data
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack()

        # Frame for the displayed processed data 
        self.data_frame = tk.Frame(self.bottom_frame)
        self.data_frame.pack(side=tk.RIGHT)

        # creat frame for the process data button
        self.data_label = tk.Label(self.data_frame, text="Processed Data:")
        self.data_label.pack()

        # creat frame for the Download Data button
        self.download_button = tk.Button(self.data_frame, text="Download Data", command=self.download_data, bg="green", fg="white")
        self.download_button.pack()

        # Adjusment function to the width and height for CSV display
        self.data_text = tk.Text(self.data_frame, height=20, width=100)
        self.data_text.pack()





    # creat an function that allows you to select files in an different path
    def select_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            self.file_path = file_path
            self.file_label.config(text=f"Selected file: {file_path}")


    # creat a function that allowes you to download the displayed labled data 
    def download_data(self):
        if self.file_path:
            labeled_data = pd.read_csv('labeled_data.csv')
            save_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])
            if save_path:
                labeled_data.to_csv(save_path, index=False)




    # read from the user-input CSV file
    # all the following code below are funtion and features from other steps. 
    # they are incorprated into the desk app to creat an function learning model GUI
    # apply MA filtering to reduce noise
    def apply_moving_average_filter(self, data, window_size=5):
        filtered_data = uniform_filter1d(data, size=window_size, axis=0, mode='nearest')
        return filtered_data

    # extract the features for futher processing
    def extract_features(self, data):
        features = {}
        combined_data = data.flatten()
        features['max'] = combined_data.max()
        features['min'] = combined_data.min()
        features['range'] = combined_data.ptp()
        features['mean'] = combined_data.mean()
        features['median'] = np.median(combined_data)
        features['variance'] = combined_data.var()
        features['skewness'] = pd.Series(combined_data).skew()
        features['std'] = combined_data.std()
        for p in [25, 50, 75]:
            features[f'percentile_{p}'] = np.percentile(combined_data, p)
        return features

    # Normalizing features using Z-score standardization 
    def normalize_features(self, features):
        normalized_features = {}
        for key, value in features.items():
            if 'mean' in key or 'std' in key: 
                normalized_features[key] = value
            else:
                feature_mean = features['mean']
                feature_std = features['std']
                normalized_value = (value - feature_mean) / feature_std if feature_std else 0
                normalized_features[key] = normalized_value
        return normalized_features





    # The function that process the overall data
    def process_data(self):
        if self.file_path:
            # read the csv file
            data = pd.read_csv(self.file_path, usecols=[4])

            # Apply moving average filter, segmenting the data, extracting the features, normalizing the featureas, and labled the data
            smoothed_data = self.apply_moving_average_filter(data.values)

            # Define segment length
            segment_length = 500
            segments = [smoothed_data[i:i + segment_length] for i in range(0, len(smoothed_data), segment_length)]

            # Extract features from segments
            features = [self.extract_features(segment) for segment in segments]
            normalized_features = [self.normalize_features(feature) for feature in features]

            # Create DataFrame uding the processed normalized features
            normalized_features_df = pd.DataFrame(normalized_features)
            normalized_features_df = normalized_features_df.drop(normalized_features_df.columns[0], axis=1)
            print(normalized_features_df.head())

            # Read the labeled train data which will be used for accuracy prediction
            # Separate features and labels for training
            train_data = pd.read_csv('train_labeled.csv', index_col=0)
            print(train_data.head())
            X_train = train_data.drop(columns=['label'])  
            y_train = train_data['label'] 

            # Perform logistic regression model and fit the model
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)

            # Make predictions on the labeled test and calculate accuracy on the labeled test data
            # Convert predictions NumPy array to a DataFrame
            # Concatenate normalized_features_df and predictions_df along axis 1
            predictions = model.predict(normalized_features_df)
            predictions_df = pd.DataFrame({'Prediction': predictions})
            labeled_data = pd.concat([normalized_features_df, predictions_df], axis=1)




            # The section below contains plot that heps user visulaize the data
            # the first plot is an seneory data plot that displayes the Z accerlation from the data
            # the second graph plots the mean accleration 
            # The seconf plot reprsent an pi chart that display an overall percentage of walk and jump -
            #---- which can be use to test accuracy from first graph.

            # Plotting the overall sensory data
            self.ax.clear()
            self.ax.set_title('STEP1: Collected Sensory Data')
            self.ax.set_xlabel('Time')
            self.ax.set_ylabel('z-axis accleration (m/s^2)')
            self.ax.plot(data.index, smoothed_data, label='Sensor Data')
            self.ax.legend()
            self.canvas.draw()

            # Create a plot that displayes the mean accerlation to viulizae which portion is walking or jumping
            fig, ax1 = plt.subplots(figsize=(8, 6))
            ax1.set_title('STEP2: Extracted Mean Value')
            ax1.set_xlabel('Time')
            ax1.set_ylabel('z-axis accleration (m/s^2)')
            ax1.plot(labeled_data.index, labeled_data['mean'], label='mean', color='darkred')
            ax1.legend()
            ax1.grid(True)
 
            # Create a new frame for the pie chart
            self.pie_chart_frame = tk.Frame(self.bottom_frame)
            self.pie_chart_frame.pack(side=tk.RIGHT)

            # Create a pie chart that displayed perctage of walk and jump
            above_10 = (labeled_data['mean'] > 10).sum()
            below_10 = (labeled_data['mean'] <= 10).sum()
            sizes = [above_10, below_10]
            labels = ['Jump', 'Walk']
            fig_pie, ax_pie = plt.subplots(figsize=(8, 6))
            ax_pie.set_title('STEP3: Percentage of walk and jump')
            ax_pie.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
            ax_pie.axis('equal')
          

            # Place the pie chart in the GUI
            canvas_pie = FigureCanvasTkAgg(fig_pie, master=self.pie_chart_frame)
            canvas_pie.draw()
            canvas_pie.get_tk_widget().pack()

            # Place the new subplot in the GUI
            new_canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
            new_canvas.draw()
            new_canvas.get_tk_widget().pack(side=tk.LEFT)

            # Save labeled data to CSV
            labeled_data.to_csv('labeled_data.csv', index=False)

            # Update displayed processed data
            self.data_text.delete(1.0, tk.END)
            self.data_text.insert(tk.END, labeled_data.to_string())


         


def main():
    root = tk.Tk()
    app = SensorDataLabeler(root)

    # creat the process button that process the data on command
    process_button = tk.Button(root, text="Process CSV", command=app.process_data, bd=3, relief=tk.RAISED, bg="red", fg="white")
    process_button.pack()
    process_button.place(x=10, y=10)

    root.mainloop()


if __name__ == "__main__":
    main()
