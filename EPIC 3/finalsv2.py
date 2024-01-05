import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

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
best_scaler_params = {'with_mean': True, 'with_std': True}

cvstruncated = TruncatedSVD(n_components=best_n_components)
user_feature_matrix_truncated = cvstruncated.fit_transform(TfidfVectorizer(**best_vectorizer_params).fit_transform(df['user_features']))

tdiffvectozer = StandardScaler(**best_scaler_params)
user_feature_matrix_normalized = tdiffvectozer.fit_transform(user_feature_matrix_truncated)

user_similarity = cosine_similarity(user_feature_matrix_normalized)

def get_recommendations(campaign_id, user_similarity, df, num_recommendations=5, content_based=False):
    campaign_df = df[df['Campagne'] == campaign_id]
    
    registered_user_indices = campaign_df.index.tolist()
    
    if not registered_user_indices:
        return [], 0, 0, 0
    
    recommended_users = []
    
    if content_based:
        user_columns = [
            'Functie_naam', 'Fucntietitel', 'Status', 'Aanwezig_Afwezig', 'Bron', 'Activiteitstype',
            'Product', 'Thema_Naam_', 'Thema', 'KeyPhrases', 'Ondernemingsaard', 'Ondernemingstype',
            'Primaire_activiteit'
        ]
        user_features = df[user_columns].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
        tfidf_vectorizer_user = TfidfVectorizer(**best_vectorizer_params)
        user_feature_matrix = tfidf_vectorizer_user.fit_transform(user_features)
        
        user_similarity = cosine_similarity(user_feature_matrix, user_feature_matrix)
        
        # Check bounds before accessing indices
        valid_indices = [i for i in registered_user_indices if i < user_similarity.shape[1]]
        if not valid_indices:
            return [], 0, 0, 0
        
        # Calculate the average similarity for each registered user
        average_similarity = np.mean(user_similarity[:, valid_indices], axis=1)
    else:
        # Check bounds before accessing indices
        valid_indices = [i for i in registered_user_indices if i < user_similarity.shape[0]]
        if not valid_indices:
            return [], 0, 0, 0
        
        # Calculate the average similarity for each registered user
        average_similarity = np.mean(user_similarity[valid_indices, :], axis=0)
    
    sorted_users = np.argsort(average_similarity)[::-1]
    
    new_recommendations = [(user_index, average_similarity[user_index]) for user_index in sorted_users if user_index not in valid_indices]
    
    recommended_users.extend(new_recommendations)
    
    # Convert indices to user IDs
    recommended_user_ids = [df.iloc[user_index]['Contactfiche'] for user_index, _ in recommended_users]
    
    # Evaluate the model using attended users as true labels
    attended_users = np.array(df[df['Campagne'] == campaign_id].index)
    
    # Filter out indices that are out of bounds
    valid_attended_users = [idx for idx in attended_users if idx < user_similarity.shape[0]]
    
    # Simulate predictions using cosine similarity scores
    simulated_predictions = np.mean(user_similarity[valid_attended_users, :], axis=0)
    
    # Simulate true labels using attended users
    true_labels = np.zeros(user_similarity.shape[1])
    true_labels[valid_attended_users] = 1
    
    # Convert predicted probabilities to binary predictions
    binary_predictions = (simulated_predictions >= 0.5).astype(int)
    
    # Calculate additional evaluation metrics
    precision = precision_score(true_labels[valid_indices], binary_predictions[valid_indices])
    recall = recall_score(true_labels[valid_indices], binary_predictions[valid_indices])
    f1 = f1_score(true_labels[valid_indices], binary_predictions[valid_indices])
    
    return recommended_user_ids, precision, recall, f1

campaign_id_to_recommend = '2D21A437-130D-EA11-8107-001DD8B72B62'
recommended_users, precision, recall, f1 = get_recommendations(campaign_id_to_recommend, user_similarity, df, content_based=True)

# Save recommendations to a file
output_file = 'user_recommendations.txt'
with open(output_file, 'w') as file:
    file.write(f"Recommended Users for Campaign: {campaign_id_to_recommend}\n")
    for i, user_id in enumerate(recommended_users, 1):
        file.write(f"{i}. User ID: {user_id}\n")

print(f"Recommendations saved to {output_file}")
print(f"Precision: {precision}, Recall: {recall}, F1 Score: {f1}")
