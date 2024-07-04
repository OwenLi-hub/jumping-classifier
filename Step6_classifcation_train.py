import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, RocCurveDisplay
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, precision_score, f1_score

# Read the CSV files from step 4-5
train_features = pd.read_csv('train_features.csv')
test_features = pd.read_csv('test_features.csv')

# Label the rows if the mean z-acceleration exceeds 9.81 m/s^2
def label_rows(row):
    if row['mean'] > 9.81:
        return 'jump'
    else:
        return 'walk'

# Apply the label function for each CSV file
# Save the CSV file for viewing
train_features['label'] = train_features.apply(label_rows, axis=1)
test_features['label'] = test_features.apply(label_rows, axis=1)
train_features.to_csv('train_labeled.csv', index=False)
test_features.to_csv('test_labeled.csv', index=False)


# Read the labeled train data which will be used for accuracy prediction
# Separate features and labels for training
train_data = pd.read_csv('train_labeled.csv')
X_train = train_data.drop(columns=['label'])  
y_train = train_data['label']  


# Check for missing values in the training data
# Handle missing values by removing NAN rows
missing_values = X_train.isnull().sum()
print("Missing values in training data:\n", missing_values)
X_train.dropna(inplace=True)
y_train = y_train[X_train.index]  


# Perform logistic regression model and fit the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)



# Read the labeled test data which will be used for accuracy prediction
test_labeled = pd.read_csv('test_labeled.csv')
X_test_labeled = test_labeled.drop(columns=['label'])  
y_test_labeled = test_labeled['label']  

# Map label values
label_mapping = {'jump': 0, 'walk': 1}
y_test_labeled_numeric = y_test_labeled.map(label_mapping)


# Make predictions on the labeled test and calculate accuracy on the labeled test data
predictions = model.predict(X_test_labeled)
accuracy = accuracy_score(y_test_labeled, predictions)
print("Accuracy on labeled test data:", accuracy)



# Calculate additional evaluation metrics
TN, FP, FN, TP = confusion_matrix(y_test_labeled_numeric, predictions).ravel()
auc = roc_auc_score(y_test_labeled_numeric, model.predict_proba(X_test_labeled)[:, 1])
sensitivity = TP / (TP + FN)
recall = recall_score(y_test_labeled_numeric, predictions)
specificity = TN / (TN + FP)
precision = precision_score(y_test_labeled_numeric, predictions)
f1 = f1_score(y_test_labeled_numeric, predictions)
fpr = FP / (FP + TN)

print("True Negative:", TN)
print("False Positive:", FP)
print("False Negative:", FN)
print("True Positive:", TP)
print("AUC:", auc)
print("Sensitivity:", sensitivity)
print("Recall:", recall)
print("Specificity:", specificity)
print("Precision:", precision)
print("F1 Score:", f1)
print("False Positive Rate:", fpr)
# Map label values for predictions
y_pred_numeric = pd.Series(predictions).map(label_mapping)

# Plot Confusion Matrix
cm = confusion_matrix(y_test_labeled_numeric, y_pred_numeric)
cm_display = ConfusionMatrixDisplay(cm).plot()
plt.title('Confusion Matrix')
plt.show()

# Plot ROC Curve
y_prob = model.predict_proba(X_test_labeled)
fpr, tpr, _ = roc_curve(y_test_labeled_numeric, y_prob[:, 1], pos_label=1)
roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot(color='red')
plt.title('ROC Curve')
plt.show()

