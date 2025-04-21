import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

# Load the Iris dataset
iris = load_iris()
data = iris.data                      # Feature values (4 columns)
labels = iris.target                  # Labels (0, 1, 2)
label_names = iris.target_names       # Actual species names (setosa, versicolor, virginica)

# Create a DataFrame for better readability
iris_df = pd.DataFrame(data, columns=iris.feature_names)

# Apply PCA to reduce dimensions from 4 to 2
pca = PCA(n_components=2)
data_reduced = pca.fit_transform(data)  # Returns transformed data with 2 principal components

# Create DataFrame of reduced dimensions
reduced_df = pd.DataFrame(data_reduced, columns=['Principal Component 1', 'Principal Component 2'])
reduced_df['Label'] = labels

# Visualize the reduced data in a 2D scatter plot
plt.figure(figsize=(8, 6))
colors = ['r', 'g', 'b']

for i, label in enumerate(np.unique(labels)):
    subset = reduced_df[reduced_df['Label'] == label]
    plt.scatter(subset['Principal Component 1'], subset['Principal Component 2'],
                label=label_names[label], color=colors[i])

plt.title('PCA on Iris Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid(True)
plt.show()
