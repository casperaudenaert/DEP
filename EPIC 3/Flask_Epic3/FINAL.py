import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

df = pd.read_csv('file.csv')
total_rows = len(df)

df.dropna(axis=0, inplace=True)

user_columns = [
    'Functie_naam', 'Fucntietitel', 'Status', 'Aanwezig_Afwezig', 'Bron', 'Activiteitstype', 'Product',
    'Thema_Naam_', 'Thema', 'KeyPhrases', 'Ondernemingsaard', 'Ondernemingstype', 'Primaire_activiteit'
]

df['user_features'] = df[user_columns].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

best_n_components = 40
best_vectorizer_params = {'max_features': 10000, 'ngram_range': (1, 1)}
best_scaler_params= {'with_mean': True, 'with_std': True}
# Now you can use the best_n_components in your TruncatedSVD model
cvstruncated = TruncatedSVD(n_components=best_n_components)
user_feature_matrix_truncated = cvstruncated.fit_transform(TfidfVectorizer(**best_vectorizer_params).fit_transform(df['user_features']))

# cvstruncated = TruncatedSVD(n_components=50)
# user_feature_matrix_truncated = cvstruncated.fit_transform(TfidfVectorizer().fit_transform(df['user_features']))

tdiffvectozer = StandardScaler(**best_scaler_params)
user_feature_matrix_normalized = tdiffvectozer.fit_transform(user_feature_matrix_truncated)

user_similarity = cosine_similarity(user_feature_matrix_normalized)

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
        tfidf_vectorizer_campaign = TfidfVectorizer(**best_vectorizer_params)
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
print('Confusion Matrix : \n' + str(confusion_matrix(true_labels,binary_predictions)))
