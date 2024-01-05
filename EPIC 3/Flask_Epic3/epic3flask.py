from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from datetime import datetime
import time
import urllib.parse
from sqlalchemy import create_engine, text
from sklearn.model_selection import train_test_split

app = Flask(__name__)


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

best_n_components = 40
best_vectorizer_params = {'max_features': 10000, 'ngram_range': (1, 1)}
best_scaler_params= {'with_mean': True, 'with_std': True}

cvstruncated = TruncatedSVD(n_components=best_n_components)
user_feature_matrix_truncated = cvstruncated.fit_transform(TfidfVectorizer(**best_vectorizer_params).fit_transform(df['user_features']))

tdiffvectozer = StandardScaler(**best_scaler_params)
user_feature_matrix_normalized = tdiffvectozer.fit_transform(user_feature_matrix_truncated)

user_similarity = cosine_similarity(user_feature_matrix_normalized)

def get_recommendations(user_id, user_similarity, df, num_recommendations=5, content_based=False):
    user_df = df[df['Contactfiche'] == user_id]
    print('started')
    registered_campaign_indices = user_df.index.tolist()
    
    if not registered_campaign_indices:
        return []
    
    recommended_campaigns = []
    
    if content_based:
        campaign_columns = [
            'Einddatum', 'Naam_in_email',
            'Startdatum', 'Type_campagne',
            'Soort_Campagne', 'Thema_Naam_'
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

print(user_similarity)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommendations', methods=['POST'])
def get_recommendations_route():
    user_id = request.form['userId']
    num_recommendations = int(request.form['numRecommendations'])
    print(f'started with {user_id} and {num_recommendations}')
    # Record the start time
    start_time = time.time()

    # Simulate a time-consuming task
    time.sleep(2)
    #df = pd.read_csv('file.csv')

    recommended_campaigns = get_recommendations(user_id, user_similarity, df, num_recommendations, content_based=True)

    unique_campaigns = set()
    filtered_recommendations = []

    for campaign_index, _ in recommended_campaigns:
        campaign_id = df.iloc[campaign_index]['Campagne']
        end_date = df.iloc[campaign_index]['Einddatum']
        campaign_name = df.iloc[campaign_index]['Naam_in_email']
        thema = df.iloc[campaign_index]['Thema_Naam_']

        # Check if the end date is in the future
        if pd.to_datetime(end_date) > datetime.now():
            if campaign_id not in unique_campaigns:
                filtered_recommendations.append((campaign_name, end_date, campaign_id, thema))
                unique_campaigns.add(campaign_id)

    # Sort recommendations by end date in ascending order
    filtered_recommendations = sorted(filtered_recommendations, key=lambda x: pd.to_datetime(x[1]))

    # Record the end time
    end_time = time.time()

    # Calculate the duration
    duration = end_time - start_time
    print(f"Time taken to calculate recommendations: {duration} seconds")
    if num_recommendations == 0:
        return render_template('recommendations.html', user_id=user_id, recommendations=filtered_recommendations)
    else:
        return render_template('recommendations.html', user_id=user_id, recommendations=filtered_recommendations[:num_recommendations])

       

if __name__ == '__main__':
    app.run(debug=True)
