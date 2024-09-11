import pandas as pd

# Load the testing data to get the correct order of IDs
testing_data = pd.read_csv('data/testing_data.csv')
testing_ids = testing_data['id'].astype(str)
# Load the test predictions
test_predictions = pd.read_csv('test_predictions.csv')

# Create a new dataframe with the correct order and only the 'labels' column
results = pd.DataFrame({'id': testing_ids})
results = results.merge(test_predictions[['id', 'predicted_label']], on='id', how='left')
results = results.rename(columns={'predicted_label': 'labels'})

# Save the results to a new CSV file
results.to_csv('results.csv', index=False)

print("Results saved to results.csv")