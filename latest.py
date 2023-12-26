import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import numpy as np

# Load your data
df = pd.read_csv('file.csv')
total_rows = len(df)

# Drop rows with missing values
df.dropna(axis=0, inplace=True)

# Concatenate user and campaign features into a single string for each entry
user_columns = ['Functie_naam', 'Fucntietitel', 'Status', 'Aanwezig_Afwezig', 'Bron', 'Activiteitstype', 'Product', 'Thema_Naam_', 'Thema', 'KeyPhrases', 'Ondernemingsaard', 'Ondernemingstype', 'Primaire_activiteit']
campaign_columns = ['Einddatum', 'Naam_in_email', 'Startdatum', 'Type_campagne', 'Soort_Campagne']
df['user_campaign_features'] = df[user_columns + campaign_columns].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

# Use TF-IDF to vectorize user and campaign features
tfidf_vectorizer = TfidfVectorizer()
feature_matrix = tfidf_vectorizer.fit_transform(df['user_campaign_features'])

# Manual grid search for Truncated SVD parameters
best_score = -np.inf
best_params = None

for n_components in [50, 100, 150]:
    for algorithm in ['randomized', 'arpack']:
        svd = TruncatedSVD(n_components=n_components, algorithm=algorithm)
        reduced_feature_matrix = svd.fit_transform(feature_matrix)
        
        # Normalize the reduced TF-IDF vectors
        normalized_feature_matrix = normalize(reduced_feature_matrix)
        
        # Calculate cosine similarity between user and campaign features
        user_campaign_similarity = cosine_similarity(normalized_feature_matrix)
        
        # Assuming user_id_to_recommend is defined
        user_id_to_recommend = '25819DD0-28CE-E811-80F7-001DD8B72B61'
        user_unseen_campaigns = df[~df['Campagne'].isin(df[df['Contactfiche'] == user_id_to_recommend]['Campagne'])]['Campagne'].unique()
        
        # Use content-based similarity for scoring
        user_index = df[df['Contactfiche'] == user_id_to_recommend].index[0]
        user_campaign_similarity_user = user_campaign_similarity[user_index, :]
        content_based_scores = user_campaign_similarity_user[df['Campagne'].isin(user_unseen_campaigns)]

        # Calculate a simple score (you may define your own scoring metric)
        score = np.mean(content_based_scores)
        
        # Update best parameters if the current score is better
        if score > best_score:
            best_score = score
            best_params = {'n_components': n_components, 'algorithm': algorithm}

# Use the best parameters in TruncatedSVD
svd = TruncatedSVD(n_components=best_params['n_components'], algorithm=best_params['algorithm'])
reduced_feature_matrix = svd.fit_transform(feature_matrix)

# Normalize the reduced TF-IDF vectors
normalized_feature_matrix = normalize(reduced_feature_matrix)

# Calculate cosine similarity between user and campaign features
user_campaign_similarity = cosine_similarity(normalized_feature_matrix)

# Make recommendations for a specific user
user_id_to_recommend = '25819DD0-28CE-E811-80F7-001DD8B72B61'
user_unseen_campaigns = df[~df['Campagne'].isin(df[df['Contactfiche'] == user_id_to_recommend]['Campagne'])]['Campagne'].unique()

# Use content-based similarity for scoring
user_index = df[df['Contactfiche'] == user_id_to_recommend].index[0]
user_campaign_similarity_user = user_campaign_similarity[user_index, :]
content_based_scores = user_campaign_similarity_user[df['Campagne'].isin(user_unseen_campaigns)]

# Get indices of top N campaigns with the highest content-based scores
top_n_indices = np.argsort(content_based_scores)[::-1]

# Filter out campaigns that the user has already seen
top_n_indices = [index for index in top_n_indices if index in user_unseen_campaigns]

# Get all available recommendations
recommended_campaigns = user_unseen_campaigns[top_n_indices]

# Print the recommendations
print(f"Recommended Campaigns for User: {user_id_to_recommend}")
for i, campaign_id in enumerate(recommended_campaigns, 1):
    end_date = df[df['Campagne'] == campaign_id]['Einddatum'].values[0]
    print(f"{i}. Campaign ID: {campaign_id}, End Date: {end_date}")


