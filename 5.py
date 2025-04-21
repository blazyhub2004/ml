# import numpy as np
# from sklearn.neighbors import KNeighborsClassifier
# import matplotlib.pyplot as plt

# # Step 1: Generate 100 random values in the range [0, 1]
# np.random.seed(42)  # For reproducibility
# x = np.random.rand(100).reshape(-1, 1)

# # Step 2: Label first 50 values
# y = np.array([1 if val <= 0.5 else 2 for val in x[:50].flatten()])  # Class1:1, Class2:2

# # Step 3: Define training and test sets
# x_train, y_train = x[:50], y
# x_test = x[50:]

# # Step 4: Classify using KNN for different k values
# k_values = [1, 2, 3, 4, 5, 20, 30]
# print("KNN Classification Results:")
# for k in k_values:
#     knn = KNeighborsClassifier(n_neighbors=k)
#     knn.fit(x_train, y_train)
#     y_pred = knn.predict(x_test)

#     print(f"\nk = {k}")
#     for i, val in enumerate(x_test.flatten()):
#         print(f"x{50+i+1:.0f} = {val:.2f} → Predicted Class: {'Class1' if y_pred[i]==1 else 'Class2'}")
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Generate 100 random values in [0, 1]
data = np.random.rand(100)

# Label first 50 points: Class1 if <= 0.5 else Class2
labels = ["Class1" if x <= 0.5 else "Class2" for x in data[:50]]

# Euclidean distance (since it's 1D, it's just absolute difference)
def euclidean_distance(x1, x2):
    return abs(x1 - x2)

# KNN implementation
def knn_classifier(train_data, train_labels, test_point, k):
    distances = [(euclidean_distance(test_point, train_data[i]), train_labels[i])
                 for i in range(len(train_data))]
    distances.sort(key=lambda x: x[0])
    k_nearest_neighbors = distances[:k]
    k_nearest_labels = [label for _, label in k_nearest_neighbors]
    return Counter(k_nearest_labels).most_common(1)[0][0]

# Split data
train_data = data[:50]
train_labels = labels
test_data = data[50:]

k_values = [1, 2, 3, 4, 5, 20, 30]
results = {}

# Classification results
print("--- k-Nearest Neighbors Classification ---")
print("Training dataset: First 50 points labeled based on the rule (x <= 0.5 → Class1, x > 0.5 → Class2)")
print("Testing dataset: Remaining 50 points to be classified\n")

for k in k_values:
    print(f"Results for k = {k}:")
    classified_labels = [knn_classifier(train_data, train_labels, test_point, k)
                         for test_point in test_data]
    results[k] = classified_labels
    for i, label in enumerate(classified_labels, start=51):
        print(f"Point x{i} (value: {test_data[i - 51]:.4f}) is classified as {label}")
    print("\n")

# Visualization
print("Classification complete.\n")

for k in k_values:
    classified_labels = results[k]
    class1_points = [test_data[i] for i in range(len(test_data)) if classified_labels[i] == "Class1"]
    class2_points = [test_data[i] for i in range(len(test_data)) if classified_labels[i] == "Class2"]

    plt.figure(figsize=(10, 6))
    plt.scatter(train_data, [0] * len(train_data),
                c=["blue" if label == "Class1" else "red" for label in train_labels],
                label="Training Data", marker="o")
    plt.scatter(class1_points, [1] * len(class1_points), c="blue", label="Class1 (Test)", marker="x")
    plt.scatter(class2_points, [1] * len(class2_points), c="red", label="Class2 (Test)", marker="x")

    plt.title(f"k-NN Classification Results for k = {k}")
    plt.xlabel("Data Points")
    plt.ylabel("Classification Level")
    plt.legend()
    plt.grid(True)
    plt.show()
