import pandas as pd

def find_s_algorithm(file_path):
    data = pd.read_csv(file_path)
    print("Training Data:\n", data)

    attributes = data.columns[:-1]
    class_label = data.columns[-1]

    hypothesis = ['?' for _ in attributes]

    for index, row in data.iterrows():
        if row[class_label].strip().lower() == 'yes':
            for i, value in enumerate(row[attributes]):
                if hypothesis[i] == '?' or hypothesis[i] == value:
                    hypothesis[i] = value
                else:
                    hypothesis[i] = '?'
    return hypothesis

file_path = './4.csv'

final_hypothesis = find_s_algorithm(file_path)

print("\nFinal Hypothesis:", final_hypothesis)
