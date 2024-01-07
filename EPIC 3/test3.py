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
    cp.Naam_in_email,
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