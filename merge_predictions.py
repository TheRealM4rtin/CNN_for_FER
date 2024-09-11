import pandas as pd

# Load the CSV files
my_model_df = pd.read_csv('test_predictions1.csv')
pretrained_df = pd.read_csv('pretrained_model_predictions.csv')

# Clean up the 'id' column in my_model_df by removing the '.jpg' extension
my_model_df['id'] = my_model_df['id'].str.replace('.jpg', '')

# Convert 'id' to integer in my_model_df
my_model_df['id'] = my_model_df['id'].astype(int)

# Ensure 'id' in pretrained_df is integer
pretrained_df['id'] = pretrained_df['id'].astype(int)

# Merge the dataframes on the 'id' column
merged_df = pd.merge(my_model_df, pretrained_df, on='id', how='outer')

# Rename the columns
merged_df = merged_df.rename(columns={
    'predicted_label': 'mymodel',
    'predicted_emotion': 'pretrained_deepface'
})

# Reorder the columns
merged_df = merged_df[['id', 'mymodel', 'pretrained_deepface']]

# Sort the dataframe by 'id'
merged_df = merged_df.sort_values('id')

# Save the merged dataframe to final.csv
merged_df.to_csv('final.csv', index=False)

print("Merged data saved to final.csv")