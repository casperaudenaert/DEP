import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

df = pd.read_csv('data.csv')

user_columns = [
    'crm_Functie_Functie', 'crm_Functie_Naam',
    'crm_ContactFunctie_Contactpersoon', 'crm_ContactFunctie_Functie',
    'crm_Contact_Contactpersoon', 'crm_Contact_Account',
    'crm_Contact_Functietitel', 'crm_Contact_Persoon_ID',
    'crm_Contact_Status', 'crm_Inschrijving_Aanwezig_Afwezig',
    'crm_Inschrijving_Bron', 'crm_Inschrijving_Contactfiche',
    'crm_Sessie_Activiteitstype',
    'crm_Sessie_Product', 'crm_Sessie_Thema_Naam_',
    'crm_Afspraak_BETREFT_CONTACTFICHE_Afspraak',
    'crm_Afspraak_BETREFT_CONTACTFICHE_Thema',
    'crm_Afspraak_BETREFT_CONTACTFICHE_Subthema',
    'crm_Afspraak_BETREFT_CONTACTFICHE_Onderwerp', 'crm_Account_Account',
    'crm_Account_Ondernemingsaard', 'crm_Account_Ondernemingstype'
]

df['user_features'] = df[user_columns].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

tfidf_vectorizer_user = TfidfVectorizer()
user_feature_matrix = tfidf_vectorizer_user.fit_transform(df['user_features'])

user_similarity = cosine_similarity(user_feature_matrix)

def get_users_not_attended_campaign(campaign_id, user_similarity, df, num_similar_users=None):
    campaign_index = df[df['crm_Campagne_Campagne'] == campaign_id].index[0]
    attended_users = df[df['crm_Campagne_Campagne'] == campaign_id]['crm_Contact_Contactpersoon'].tolist()
    
    campaign_similarity = user_similarity[:, campaign_index]
    sorted_users = np.argsort(campaign_similarity)[::-1]
    
    similar_users = [(user_index, campaign_similarity[user_index]) for user_index in sorted_users if
                     user_index != campaign_index and df.iloc[user_index]['crm_Contact_Contactpersoon'] not in attended_users]
    
    unique_users = []
    recommended_user_ids = set()
    
    for user_index, similarity in similar_users:
        user_id = df.iloc[user_index]['crm_Contact_Contactpersoon']
        
        if user_id not in recommended_user_ids:
            unique_users.append((user_index, similarity))
            recommended_user_ids.add(user_id)
        
        if num_similar_users is not None and len(unique_users) >= num_similar_users:
            break
    
    return unique_users

campaign_id_to_find_users_for = '810A8FF6-4B3F-E911-80FC-001DD8B72B61' 
num_users_to_display = 20  

similar_users_to_campaign = get_users_not_attended_campaign(campaign_id_to_find_users_for, user_similarity, df, num_users_to_display)

print(f"Users Similar to Campaign {campaign_id_to_find_users_for} (Not Attended):")
for i, (user_index, similarity) in enumerate(similar_users_to_campaign, 1):
    user_id = df.iloc[user_index]['crm_Contact_Contactpersoon']
    print(f"{i}. User ID: {user_id}, Similarity: {similarity*100:.2f}%")
