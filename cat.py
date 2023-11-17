# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Load the data
# Assuming you have a CSV file with columns like 'GPA', 'Age', 'Gender', 'Enrollment_Status', 'Graduation_Status', etc.
data = pd.read_csv('your_data.csv')

# Select relevant features and target variable
features = ['GPA', 'Age', 'Gender']  # Add other relevant features
target_enrollment = 'Enrollment_Status'
target_graduation = 'Graduation_Status'

# Create feature matrix (X) and target vector (y) for enrollment
X_enrollment = data[features]
y_enrollment = data[target_enrollment]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_enrollment, y_enrollment, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the logistic regression model
logreg_model = LogisticRegression()
logreg_model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = logreg_model.predict(X_test_scaled)

# Evaluate the model
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Plot ROC curve (Receiver Operating Characteristic)
# Note: This is useful for binary classification problems
from sklearn.metrics import roc_curve, auc

y_prob = logreg_model.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()
