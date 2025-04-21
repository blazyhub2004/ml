import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# Step 1: Generate 100 random values in the range [0, 1]
np.random.seed(42)  # For reproducibility
x = np.random.rand(100).reshape(-1, 1)

# Step 2: Label first 50 values
y = np.array([1 if val <= 0.5 else 2 for val in x[:50].flatten()])  # Class1:1, Class2:2

# Step 3: Define training and test sets
x_train, y_train = x[:50], y
x_test = x[50:]

# Step 4: KNN classification for different k values
k_values = [1, 2, 3, 4, 5, 20, 30]
print("KNN Classification Results:")

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)

    print(f"\nResults for k = {k}")
    for i, val in enumerate(x_test.flatten()):
        label = "Class1" if y_pred[i] == 1 else "Class2"
        print(f"x{50 + i + 1} = {val:.2f} → Predicted Class: {label}")

    # Plotting
    plt.figure(figsize=(10, 4))
    plt.scatter(x_train[y_train == 1], [0]*sum(y_train == 1), color='blue', label='Class1 (Train)')
    plt.scatter(x_train[y_train == 2], [0]*sum(y_train == 2), color='red', label='Class2 (Train)')

    plt.scatter(x_test[y_pred == 1], [1]*sum(y_pred == 1), color='blue', marker='x', label='Class1 (Test)')
    plt.scatter(x_test[y_pred == 2], [1]*sum(y_pred == 2), color='red', marker='x', label='Class2 (Test)')

    plt.title(f'k-NN Classification (k = {k})')
    plt.xlabel('Value')
    plt.yticks([0, 1], ['Train', 'Test'])
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
