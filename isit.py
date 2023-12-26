import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix

# Load data
df = pd.read_csv('file.csv')
total_rows = len(df)

# Drop rows with missing values
df.dropna(axis=0, inplace=True)

# Define user columns
user_columns = [
    'Functie_naam', 'Fucntietitel', 'Status', 'Aanwezig_Afwezig', 'Bron', 'Activiteitstype', 'Product',
    'Thema_Naam_', 'Thema', 'KeyPhrases', 'Ondernemingsaard', 'Ondernemingstype', 'Primaire_activiteit'
]

# Combine user columns into a single feature
df['user_features'] = df[user_columns].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

# Vectorization using TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()
user_feature_matrix_tfidf = tfidf_vectorizer.fit_transform(df['user_features'])

# Hyperparameter tuning for TruncatedSVD
# Hyperparameter tuning for TruncatedSVD
param_grid_svd = {
    'n_components': [10, 20, 30, 40, 50]
}

svd = TruncatedSVD(random_state=42)

# Use a custom scoring function or 'neg_mean_squared_error' depending on your preference
scoring_function = 'accuracy'
grid_search_svd = GridSearchCV(svd, param_grid_svd, cv=5, scoring=scoring_function, verbose=1)
grid_search_svd.fit(user_feature_matrix_tfidf)

best_n_components_svd = grid_search_svd.best_params_['n_components']
svd = TruncatedSVD(n_components=best_n_components_svd, random_state=42)

user_feature_matrix_truncated = svd.fit_transform(user_feature_matrix_tfidf)
user_feature_matrix_normalized = StandardScaler().fit_transform(user_feature_matrix_truncated)
user_similarity = cosine_similarity(user_feature_matrix_normalized)

print
# Recommendation function
def get_recommendations(user_id, user_similarity, df, num_recommendations=5, content_based=False):
    user_df = df[df['Contactfiche'] == user_id]
    
    registered_campaign_indices = user_df.index.tolist()
    
    if not registered_campaign_indices:
        return []
    
    recommended_campaigns = []
    
    if content_based:
        campaign_columns = [
            'Einddatum', 'Naam_in_email',
            'Startdatum', 'Type_campagne',
            'Soort_Campagne'
        ]
        campaign_features = df[campaign_columns].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
        tfidf_vectorizer_campaign = TfidfVectorizer()
        campaign_feature_matrix = tfidf_vectorizer_campaign.fit_transform(campaign_features)
        
        campaign_similarity = cosine_similarity(campaign_feature_matrix)
        
        average_similarity = np.mean(campaign_similarity[registered_campaign_indices], axis=0)
    else:
        average_similarity = np.mean(user_similarity[registered_campaign_indices], axis=0)
    
    sorted_campaigns = np.argsort(average_similarity)[::-1]
    
    new_recommendations = [(campaign_index, average_similarity[campaign_index]) for campaign_index in sorted_campaigns if campaign_index not in registered_campaign_indices]
    
    recommended_campaigns.extend(new_recommendations)
    
    return recommended_campaigns

user_id_to_recommend = '25819DD0-28CE-E811-80F7-001DD8B72B61'
recommended_campaigns = get_recommendations(user_id_to_recommend, user_similarity, df, content_based=True)

unique_campaigns = set()
filtered_recommendations = []

for campaign_index, probability in recommended_campaigns:
    campaign_id = df.iloc[campaign_index]['Campagne']
    if campaign_id not in unique_campaigns:
        filtered_recommendations.append((campaign_index, probability))
        unique_campaigns.add(campaign_id)

# Print recommendations
output_file = 'recommendations.txt'
with open(output_file, 'w') as file:
    file.write(f"Recommended Campaigns for User: {user_id_to_recommend}\n")
    for i, (campaign_index, probability) in enumerate(filtered_recommendations, 1):
        campaign_id = df.iloc[campaign_index]['Campagne']
        end_date = df.iloc[campaign_index]['Einddatum']
        file.write(f"{i}. Campaign ID: {campaign_id}, Probability: {probability*100:.2f}%, End Date: {end_date}\n")

print(f"Recommendations saved to {output_file}")

# Evaluate the model using attended campaigns as true labels
attended_campaigns = df[df['Contactfiche'] == user_id_to_recommend].index.tolist()

# Simulate predictions using TfidfVectorizer-based similarity scores
simulated_predictions = np.mean(user_similarity[attended_campaigns], axis=0)

# Simulate true labels using attended campaigns
true_labels = np.zeros(len(df))
true_labels[attended_campaigns] = 1

# Convert predicted probabilities to binary predictions
binary_predictions = (simulated_predictions >= 0.5).astype(int)

# Calculate additional evaluation metrics
precision = precision_score(true_labels, binary_predictions)
recall = recall_score(true_labels, binary_predictions)
f1 = f1_score(true_labels, binary_predictions)

# Print additional evaluation metrics
print("Additional Evaluation Metrics:")
print(f"Precision: {precision}, Recall: {recall}, F1 Score: {f1}")
from sklearn.metrics import confusion_matrix
print('Confusion Matrix : \n' + str(confusion_matrix(true_labels, binary_predictions)))
