# Logistic Regression on Iris Dataset

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib


# 1. Load the Iris dataset
iris = load_iris()

X = iris.data
y = iris.target

print("Dataset loaded successfully")
print("Number of samples:", X.shape[0])
print("Number of features:", X.shape[1])


# 2. Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# 3. Create the Logistic Regression model
model = LogisticRegression(max_iter=200)

# 4. Train the model
model.fit(X_train, y_train)

print("\nModel training completed")


# 5. Make predictions
y_pred = model.predict(X_test)


# 6. Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=iris.target_names
))


# 7. Save the trained model
joblib.dump(model, "models/iris_logistic_regression.pkl")

print("\nModel saved successfully!")