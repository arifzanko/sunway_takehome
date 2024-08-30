import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
dataset = pd.read_csv('datasets.csv')

# Separate features and target
X = dataset.drop(columns=['sentiment'])  # Features
y = dataset['sentiment']  # Target

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

# Initialize and train Decision Tree Classifier
clf = DecisionTreeClassifier(ccp_alpha=0.01)
clf = clf.fit(X_train, y_train)

# Make predictions on the test set
predictions = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)

# Save the trained model
joblib.dump(clf, 'decision_tree_model.joblib')

# Print results
print('Prediction: ',predictions)
print('Accuracy: ',accuracy)