import pandas as pd
from surprise import Reader, Dataset
from surprise.model_selection import train_test_split
from surprise import KNNBasic
from surprise import accuracy
from sklearn.metrics import average_precision_score, mean_squared_error, mean_absolute_error, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
from sklearn.preprocessing import binarize

# Load the data
df = pd.read_csv('test.csv')
full = df
df = df[['crm_Contact_Contactpersoon', 'crm_Campagne_Campagne', 'Similarity', 'crm_Campagne_Naam_in_email']]

# Load data into Surprise
reader = Reader(rating_scale=(0, 10))
data = Dataset.load_from_df(df[['crm_Contact_Contactpersoon', 'crm_Campagne_Campagne', 'Similarity']], reader)

# Split the data into train and test sets
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# Train the model
algo = KNNBasic(sim_options={
    "name": "cosine",
    "user_based": False,
})
algo.fit(trainset)

# Specify the user ID
user_id = '01B45481-0877-E911-80FE-001DD8B72B62'

# Get the campaigns that the user has visited
user_campaigns = df[df['crm_Contact_Contactpersoon'] == user_id]['crm_Campagne_Campagne'].unique()

# Create a testset with all campaigns (including those visited by the user)
full_testset = [(user_id, campaign, 0) for campaign in user_campaigns] + list(testset)

# Make predictions
predictions = algo.test(full_testset)

# Extract predicted ratings
predicted_ratings = [pred.est for pred in predictions]

# Calculate metrics
ground_truth = [pred.r_ui for pred in predictions]

# Calculate MAP
y_true = [1 if campaign in user_campaigns else 0 for (_, campaign, _) in full_testset]
y_score = predicted_ratings
map_score = average_precision_score(y_true, y_score)

# Calculate MSE, RMSE, and MAE
mse = mean_squared_error(ground_truth, predicted_ratings)
rmse = np.sqrt(mse)
mae = mean_absolute_error(ground_truth, predicted_ratings)

# Binarize the ground truth for classification metrics
# Binarize the ground truth for classification metrics
threshold = 5  # You can adjust the threshold based on your preference
y_true_binary = np.where(np.array(ground_truth) >= threshold, 1, 0)
y_pred_binary = np.where(np.array(predicted_ratings) >= threshold, 1, 0)


# Calculate precision, recall, f1-score, and ROC AUC
precision = precision_score(y_true_binary, y_pred_binary)
recall = recall_score(y_true_binary, y_pred_binary)
f1 = f1_score(y_true_binary, y_pred_binary)
roc_auc = roc_auc_score(y_true_binary, predicted_ratings)

# Print all metrics
print(f"Mean Average Precision (MAP): {map_score}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1}")
print(f"ROC AUC: {roc_auc}")

# Rest of your code...
