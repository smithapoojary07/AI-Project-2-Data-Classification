# ============================================================
# ARTIFICIAL INTELLIGENCE - PROJECT 2
# DATA CLASSIFICATION USING AIS
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report


# ============================================================
# 1. LOAD AND UNDERSTAND THE DATASET
# ============================================================

iris = load_iris()

df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

df["target"] = iris.target

print("First 5 rows of the dataset:")
print(df.head())

print("\nDataset Information:")
df.info()

print("\nDataset Shape:")
print(df.shape)

print("\nTarget Classes:")
print(iris.target_names)


# ============================================================
# 2. SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop("target", axis=1)
y = df["target"]

print("\nFeatures:")
print(X.head())

print("\nTarget:")
print(y.head())


# ============================================================
# 3. SPLIT DATA INTO TRAINING AND TESTING SETS
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining data size:", X_train.shape)
print("Testing data size:", X_test.shape)


# ============================================================
# 4. APPLY A SIMPLE CLASSIFICATION ALGORITHM
# ============================================================

model = DecisionTreeClassifier(random_state=42)

model.fit(X_train, y_train)

print("\nDecision Tree model trained successfully!")


# ============================================================
# 5. MAKE PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)

print("\nPredicted values:")
print(y_pred)

print("\nActual values:")
print(y_test.values)


# ============================================================
# 6. CHECK MODEL ACCURACY
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(accuracy)

print("\nAccuracy Percentage:")
print(accuracy * 100, "%")


# ============================================================
# 7. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=iris.target_names
    )
)


# ============================================================
# 8. VISUALIZE CLASSIFICATION RESULTS
# ============================================================

plt.figure(figsize=(8, 5))

plt.scatter(
    X_test.iloc[:, 0],
    X_test.iloc[:, 1],
    c=y_pred
)

plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("Iris Dataset - Classification Results")

plt.show()


# ============================================================
# 9. TEST THE MODEL WITH NEW DATA
# ============================================================

new_data = [[5.1, 3.5, 1.4, 0.2]]

prediction = model.predict(new_data)

print("\nPrediction for New Data:")
print("Predicted Class:", iris.target_names[prediction[0]])
