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

def get_recommendations(campaign_id, user_similarity, df, num_recommendations=5, content_based=False):
    campaign_df = df[df['Campagne'] == campaign_id]
    
    registered_user_indices = campaign_df.index.tolist()
    
    if not registered_user_indices:
        return []
    
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
            return []
        
        # Calculate the average similarity for each registered user
        average_similarity = np.mean(user_similarity[:, valid_indices], axis=1)
    else:
        # Check bounds before accessing indices
        valid_indices = [i for i in registered_user_indices if i < user_similarity.shape[1]]
        if not valid_indices:
            return []
        
        # Calculate the average similarity for each registered user
        average_similarity = np.mean(user_similarity[:, valid_indices], axis=1)
    
    sorted_users = np.argsort(average_similarity)[::-1]
    
    new_recommendations = [(user_index, average_similarity[user_index]) for user_index in sorted_users if user_index not in valid_indices]
    
    recommended_users.extend(new_recommendations)
    
    return recommended_users

campaign_id_to_recommend = '15F31E36-F409-E911-80FA-001DD8B72B62'
recommended_users = get_recommendations(campaign_id_to_recommend, user_similarity, df, content_based=True)

# Save recommendations to a file
output_file = 'user_recommendations.txt'
with open(output_file, 'w') as file:
    file.write(f"Recommended Users for Campaign: {campaign_id_to_recommend}\n")
    for i, (user_index, probability) in enumerate(recommended_users, 1):
        user_id = df.iloc[user_index]['Contactfiche']
        file.write(f"{i}. User ID: {user_id}, Probability: {probability*100:.2f}%\n")

print(f"Recommendations saved to {output_file}")
