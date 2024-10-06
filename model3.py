import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the dataset from the CSV file
csv_file_path = 'fruit_classification_dataset.csv'  # Make sure to adjust the path to the actual CSV location
fruit_data = pd.read_csv(csv_file_path)

# Step 2: Preprocess and Split the Dataset
X = fruit_data[['Weight', 'Size', 'Color_Code']]  # Features
y = fruit_data['Fruit_Type']  # Target

# Split the dataset into training (80%) and testing sets (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Step 4: Make Predictions
y_pred = rf_classifier.predict(X_test)

# Step 5: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

print("Classification Report:")
print(classification_report(y_test, y_pred))

# Step 6: Plot feature importance
feature_importances = rf_classifier.feature_importances_
features = X.columns

# Plotting
plt.figure(figsize=(8, 6))
sns.barplot(x=feature_importances, y=features)
plt.title('Feature Importance in Fruit Classification')
plt.show()
