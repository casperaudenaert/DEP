import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

import urllib.parse
from sqlalchemy import create_engine, text



server = '127.0.0.1,1438'
database = 'Project'
username = 'sa'
password = 'root_1234'


params = urllib.parse.quote_plus(
    'DRIVER={ODBC Driver 17 for SQL Server};' +
    'SERVER=' + server + ';DATABASE=' + database + ';UID=' + username + ';PWD=' + password)

# Create the SQLAlchemy engine
engine = create_engine("mssql+pyodbc:///?odbc_connect=%s" % params)

try:
    # Establish a connection
    connection = engine.connect()
    print("Connected to the database successfully!")

    sql_query = """SELECT
	f.Functie, 
	f.Functie_naam,
	cf.Conctactpersoon,
	cf.Functie,
	c.Contactfiche,
	c.Account,
	c.Fucntietitel,
	c.Persoon,
	c.[Status],
	I.Aanwezig_Afwezig,
	I.Bron,
	I.Contactfiche,
	I.Inschrijving,
	si.Inschrijving,
	si.Sessie,
	si.SessieInschrijving,
	s.Activiteitstype,
	s.Campagne,
	s.Product,
	s.Sessie,
	s.Thema_Naam_,
	cp.Campagne,
	cp.Einddatum,
	cp.Naam_in_email,
	cp.Startdatum,
	cp.Type_campagne,
	cp.Soort_Campagne,
	abc.Afspraak,
	abc.Thema,
	abc.Betreft_id,
	abc.KeyPhrases,
	a.Account,
	a.Ondernemingsaard,
	a.Ondernemingstype,
	a.Primaire_activiteit
FROM 
	Functies f
JOIN
	Contactfiches_functies cf ON f.Functie = cf.Functie
JOIN
	Contactfiches c ON cf.Conctactpersoon = c.Contactfiche
JOIN
	Inschrijving I ON c.Contactfiche = I.Contactfiche
JOIN
	SessieInschrijving si ON I.Inschrijving = si.Inschrijving
JOIN
	sessie s ON si.Sessie = s.Sessie
JOIN
	Campagne cp ON s.Campagne = cp.Campagne
JOIN
	Afspraak_betreft_contactfiche abc ON c.Contactfiche = abc.Betreft_id
JOIN
	Accounts a ON c.Account = a.Account"""

    df = pd.read_sql(text(sql_query), connection)

    connection.close()
except Exception as e:
    print("Connection failed! Error:", e)
df = pd.read_csv('file.csv')
total_rows = len(df)
part_size = total_rows // 4

midpoint = len(df) // 2
first_half = df.loc[:midpoint-1]
second_half = df.loc[midpoint:]

df = first_half

user_columns = [
    'Functie_naam','Fucntietitel','Status', 'Aanwezig_Afwezig','Bron','Activiteitstype','Product','Thema_Naam_','Thema','KeyPhrases','Ondernemingsaard','Ondernemingstype','Primaire_activiteit'
]

df['user_features'] = df[user_columns].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

tfidf_vectorizer_user = TfidfVectorizer()
user_feature_matrix = tfidf_vectorizer_user.fit_transform(df['user_features'])

user_similarity = cosine_similarity(user_feature_matrix)

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

print("Recommended Campaigns for User:", user_id_to_recommend)
for i, (campaign_index, probability) in enumerate(filtered_recommendations, 1):
    campaign_id = df.iloc[campaign_index]['Campagne']
    print(f"{i}. Campaign ID: {campaign_id}, Probability: {probability*100:.2f}%")
