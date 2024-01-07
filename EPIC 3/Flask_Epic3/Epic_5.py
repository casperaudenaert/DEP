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

    sql_query = """SELECT TOP 5000
    f.Functie_naam,
    cf.Conctactpersoon,
    c.Contactfiche,
    c.Fucntietitel,
    c.[Status],
    I.Aanwezig_Afwezig,
    I.Bron,
    s.Activiteitstype,
    s.Product,
    s.Thema_Naam_,
    cp.Campagne,
    cp.Einddatum,
    cp.Naam_in_email,
    cp.Startdatum,
    cp.Type_campagne,
    cp.Soort_Campagne,
    abc.Thema,
    abc.KeyPhrases,
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
    df.dropna(axis=0, inplace=True)
    df.to_csv("database.csv")
    connection.close()
except Exception as e:
    print("Connection failed! Error:", e)


df = pd.read_csv('database.csv')
df.dropna(axis=0, inplace=True)

user_columns = [
    'Functie_naam', 'Fucntietitel', 'Status', 'Aanwezig_Afwezig', 'Bron', 'Activiteitstype', 'Product', 'Thema', 'KeyPhrases', 'Ondernemingsaard', 'Ondernemingstype', 'Primaire_activiteit'
]

df['user_features'] = df[user_columns].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

tfidf_vectorizer_user = TfidfVectorizer()
user_feature_matrix = tfidf_vectorizer_user.fit_transform(df['user_features'])

user_similarity = cosine_similarity(user_feature_matrix)

def get_users_not_attended_campaign(campaign_id, user_similarity, df, num_similar_users=None):
    campaign_index = df[df['Campagne'] == campaign_id].index[0]
    attended_users = df[df['Campagne'] == campaign_id]['Conctactpersoon'].tolist()
    
    campaign_similarity = user_similarity[:, campaign_index]
    sorted_users = np.argsort(campaign_similarity)[::-1]
    
    similar_users = [(user_index, campaign_similarity[user_index]) for user_index in sorted_users if
                     user_index != campaign_index and df.iloc[user_index]['Conctactpersoon'] not in attended_users]
    
    unique_users = []
    recommended_user_ids = set()
    
    for user_index, similarity in similar_users:
        user_id = df.iloc[user_index]['Conctactpersoon']
        
        if user_id not in recommended_user_ids:
            unique_users.append((user_index, similarity))
            recommended_user_ids.add(user_id)
        
        if num_similar_users is not None and len(unique_users) >= num_similar_users:
            break
    
    return unique_users

campaign_id_to_find_users_for = '810A8FF6-4B3F-E911-80FC-001DD8B72B61' 
num_users_to_display = 20  

similar_users_to_campaign = get_users_not_attended_campaign(campaign_id_to_find_users_for, user_similarity, df, num_users_to_display)

def calculate_precision_recall_f1(campaign_id, user_similarity, df, num_similar_users=None):
    campaign_index = df[df['Campagne'] == campaign_id].index[0]
    attended_users = df[df['Campagne'] == campaign_id]['Conctactpersoon'].tolist()

    campaign_similarity = user_similarity[:, campaign_index]
    sorted_users = np.argsort(campaign_similarity)[::-1]

    recommended_user_ids = set()
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for user_index in sorted_users:
        user_id = df.iloc[user_index]['Conctactpersoon']

        if user_id not in attended_users:
            if user_id not in recommended_user_ids:
                false_positives += 1
        else:
            true_positives += 1

        recommended_user_ids.add(user_id)

        if num_similar_users is not None and len(recommended_user_ids) >= num_similar_users:
            break

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1_score

precision, recall, f1_score = calculate_precision_recall_f1(campaign_id_to_find_users_for, user_similarity, df, num_users_to_display)

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1_score:.2f}")

print(f"Users Similar to Campaign {campaign_id_to_find_users_for} (Not Attended):")
for i, (user_index, similarity) in enumerate(similar_users_to_campaign, 1):
    user_id = df.iloc[user_index]['Conctactpersoon']
    print(f"{i}. User ID: {user_id}, Similarity: {similarity*100:.2f}%")
