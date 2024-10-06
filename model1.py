import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset from the CSV file
csv_file_path = 'student_attendance_performance.csv'  # Path to the CSV file
student_data = pd.read_csv(csv_file_path)

# Feature matrix X and target vector y
X = student_data[['Attendance', 'Performance', 'Participation', 'Study_Hours', 'Past_Failures']]
y = student_data['Dropout_Fail']

# Split dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize KNN classifier with k=3
knn = KNeighborsClassifier(n_neighbors=3)

# Train the model
knn.fit(X_train_scaled, y_train)

# Predict on the test data
y_pred = knn.predict(X_test_scaled)

# Model accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Adding predicted and actual values to the test dataframe
X_test['Predicted_Dropout_Fail'] = y_pred
X_test['Actual_Dropout_Fail'] = y_test.values

# Save the result to a CSV file
result_csv_file_path = 'student_attendance_performance_predictions.csv'
X_test.to_csv(result_csv_file_path, index=False)

print(f"Results have been exported to {result_csv_file_path}")
