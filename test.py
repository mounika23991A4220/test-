import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load your dataset (replace this with your actual dataset file)
# For this example, we assume a CSV file with features and a target column 'Disease'
feature_names = ["Age", "BloodPressure", "Cholesterol", "HeartRate", "Glucose"]
data = pd.read_csv("data.csv")

# Let's assume the target column is called 'Disease' (1 for diseased, 0 for not)
# The rest of the columns are medical features (e.g., age, blood pressure, etc.)

# Separating features and target variable
X = data.drop(columns='Disease')  # Features (independent variables)
y = data['Disease']  # Target (dependent variable)

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Output the results
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)
warnings.filterwarnings("ignore")

# Now, you can use the model to make predictions on new data
# Example: Predicting for a new sample (replace with actual values)
feature_names = ["Age", "BloodPressure", "Cholesterol", "HeartRate", "Glucose"]
feature_names = X.columns.tolist()
new_sample = [[45, 30, 285, 90, 200]]  # Example feature values for a new sample
prediction = model.predict(new_sample)
print(f"Prediction for new sample: {'Disease' if prediction[0] == 1 else 'No Disease'}")
