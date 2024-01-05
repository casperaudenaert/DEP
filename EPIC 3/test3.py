import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix

df = pd.read_csv('file.csv')
total_rows = len(df)

df.dropna(axis=0, inplace=True)

campaign_columns = [
    'Einddatum', 'Naam_in_email',
    'Startdatum', 'Type_campagne',
    'Soort_Campagne'
]

df['campaign_features'] = df[campaign_columns].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

best_n_components = 40
best_vectorizer_params = {'max_features': 10000, 'ngram_range': (1, 1)}
best_scaler_params = {'with_mean': True, 'with_std': True}

cvstruncated = TruncatedSVD(n_components=best_n_components)
campaign_feature_matrix_truncated = cvstruncated.fit_transform(TfidfVectorizer(**best_vectorizer_params).fit_transform(df['campaign_features']))

tdiffvectozer = StandardScaler(**best_scaler_params)
campaign_feature_matrix_normalized = tdiffvectozer.fit_transform(campaign_feature_matrix_truncated)

campaign_similarity = cosine_similarity(campaign_feature_matrix_normalized)

def get_users_for_campaign(campaign_id, campaign_similarity, df, num_recommendations=5):
    campaign_df = df[df['Campagne'] == campaign_id]
    
    if campaign_df.empty:
        return []
    
    registered_user_indices = campaign_df.index.tolist()
    
    if not registered_user_indices:
        return []
    
    # Ensure that indices are within the valid range
    registered_user_indices = [idx for idx in registered_user_indices if idx < len(campaign_similarity)]
    
    if not registered_user_indices:
        return []
    
    recommended_users = []
    
    average_similarity = np.mean(campaign_similarity[registered_user_indices], axis=0)
    
    sorted_users = np.argsort(average_similarity)[::-1]
    
    new_recommendations = [(user_index, average_similarity[user_index]) for user_index in sorted_users if user_index not in registered_user_indices]
    
    recommended_users.extend(new_recommendations)
    
    return recommended_users


campaign_id_to_search = '24F95D75-F0F7-EA11-8115-001DD8B72B62'  # Replace with the actual campaign ID
recommended_users = get_users_for_campaign(campaign_id_to_search, campaign_similarity, df)

unique_users = set()
filtered_recommendations = []

for user_index, probability in recommended_users:
    user_id = df.iloc[user_index]['Contactfiche']
    if user_id not in unique_users:
        filtered_recommendations.append((user_index, probability))
        unique_users.add(user_id)

# Print recommendations
output_file = 'user_recommendations.txt'
with open(output_file, 'w') as file:
    file.write(f"Recommended Users for Campaign: {campaign_id_to_search}\n")
    for i, (user_index, probability) in enumerate(filtered_recommendations, 1):
        if user_index < len(df):
            user_id = df.iloc[user_index]['Contactfiche']
            functie_naam = df.iloc[user_index]['Functie_naam']
            file.write(f"{i}. User ID: {user_id}, Probability: {probability*100:.2f}%, Functie Naam: {functie_naam}\n")
        else:
            print(f"Warning: User index {user_index} is out of bounds for the DataFrame.")

print(f"User recommendations saved to {output_file}")

# Evaluate the model using attended users as true labels
attended_users = df[df['Campagne'] == campaign_id_to_search].index.tolist()

# Simulate predictions using TfidfVectorizer-based similarity scores
simulated_predictions = np.mean(campaign_similarity[attended_users], axis=0)

# Simulate true labels using attended users
true_labels = np.zeros(len(df))
true_labels[attended_users] = 1

# Convert predicted probabilities to binary predictions
binary_predictions = (simulated_predictions >= 0.5).astype(int)

# Calculate additional evaluation metrics
precision = precision_score(true_labels, binary_predictions)
recall = recall_score(true_labels, binary_predictions)
f1 = f1_score(true_labels, binary_predictions)

# Print additional evaluation metrics
print("Additional Evaluation Metrics:")
print(f"Precision: {precision}, Recall: {recall}, F1 Score: {f1}")
print('Confusion Matrix : \n' + str(confusion_matrix(true_labels, binary_predictions)))
