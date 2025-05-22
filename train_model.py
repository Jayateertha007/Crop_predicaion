# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import pickle

# Load the dataset
df = pd.read_csv('Crop Recommendation.csv')

# Separate features and target label
features = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
target = df['label']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=2)

# Initialize the Decision Tree Classifier
model = DecisionTreeClassifier(criterion='entropy', random_state=2, max_depth=5)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = metrics.accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save the trained model to a .pkl file
model_filename = 'crop_model.pkl'
with open(model_filename, 'wb') as model_file:
    pickle.dump(model, model_file)

print(f"Model saved as {model_filename}")

# Optionally: Load the saved model to test its functionality
with open(model_filename, 'rb') as model_file:
    loaded_model = pickle.load(model_file)

# Example prediction with a new input (you can replace this with any values)
new_data = np.array([[20, 43, 43, 25.95, 61.89, 8.33, 99.57]])  # Example data
prediction = loaded_model.predict(new_data)
print(f"Prediction for new data: {prediction}")
