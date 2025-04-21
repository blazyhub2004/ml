import pandas as pd

def find_s_algorithm(file_path):
    # Load the dataset
    data = pd.read_csv(file_path)
    print("Training Data:\n", data)

    # Extract attributes and target (class) column
    attributes = data.columns[:-1]
    class_label = data.columns[-1]

    # Initialize hypothesis with the most specific values
    hypothesis = ['?' for _ in attributes]

    # Iterate through each row
    for index, row in data.iterrows():
        # Only consider positive examples (e.g., labeled 'Yes')
        if row[class_label].strip().lower() == 'yes':
            for i, value in enumerate(row[attributes]):
                if hypothesis[i] == '?' or hypothesis[i] == value:
                    hypothesis[i] = value
                else:
                    hypothesis[i] = '?'
    return hypothesis

# Path to your CSV file (update path if needed)
file_path = './4.csv'

# Run Find-S
final_hypothesis = find_s_algorithm(file_path)

# Output final hypothesis
print("\nFinal Hypothesis:", final_hypothesis)
