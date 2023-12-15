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

def get_recommendations(user_id, user_similarity, df, num_recommendations=5, content_based=False):
    user_df = df[df['crm_Contact_Contactpersoon'] == user_id]
    
    registered_campaign_indices = user_df.index.tolist()
    
    if not registered_campaign_indices:
        return []
    
    recommended_campaigns = []
    
    if content_based:
        campaign_columns = [
            'crm_Campagne_Einddatum', 'crm_Campagne_Naam_in_email',
            'crm_Campagne_Startdatum', 'crm_Campagne_Type_campagne',
            'crm_Campagne_Soort_Campagne'
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

user_id_to_recommend = '0B6B8265-53FC-E811-80F9-001DD8B72B61'
recommended_campaigns = get_recommendations(user_id_to_recommend, user_similarity, df, content_based=True)

unique_campaigns = set()
filtered_recommendations = []

for campaign_index, probability in recommended_campaigns:
    campaign_id = df.iloc[campaign_index]['crm_Campagne_Campagne']
    if campaign_id not in unique_campaigns:
        filtered_recommendations.append((campaign_index, probability))
        unique_campaigns.add(campaign_id)

print("Recommended Campaigns for User:", user_id_to_recommend)
for i, (campaign_index, probability) in enumerate(filtered_recommendations, 1):
    campaign_id = df.iloc[campaign_index]['crm_Campagne_Campagne']
    print(f"{i}. Campaign ID: {campaign_id}, Probability: {probability*100:.2f}%")
