import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score

# Load the dataset
file_path = 'data.csv'
data = pd.read_csv(file_path)

text_column = 'crm_Afspraak_BETREFT_CONTACTFICHE_KeyPhrases'
data[text_column] = data[text_column].fillna('')
# Define the user-related columns
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


target_column = 'crm_Campagne_Campagne'

# Preprocessing for numerical and categorical data
numeric_features = data[user_columns].select_dtypes(include=['int64', 'float64']).columns
categorical_features = data[user_columns].select_dtypes(include=['object']).columns

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

text_transformer = Pipeline(steps=[
    ('tfidf', TfidfVectorizer(tokenizer=lambda x: x.split(','))),
    ('svd', TruncatedSVD(n_components=100))  # Adjust as needed
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features),
        ('txt', text_transformer, text_column)
    ])

# Building the Gradient Boosting model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', GradientBoostingClassifier(verbose=True))
])

# Splitting data for training and testing
X = data[user_columns + [text_column]]
y = data[target_column]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Training the model
model.fit(X_train, y_train)

# Evaluate the model
cv_results = cross_validate(model, X, y, cv=5, scoring='accuracy')
print(cv_results)

# Function to recommend campaigns for a given user ID
def recommend_campaigns_for_user(user_id, model, data, user_columns, text_column, target_column, top_n=5):
    if user_id not in data['crm_Contact_Contactpersoon'].values:
        return f"No data available for user ID {user_id}"

    # Get the user's data
    user_data = data[data['crm_Contact_Contactpersoon'] == user_id]

    # Exclude campaigns the user has already signed up for
    signed_up_campaigns = set(user_data[target_column].unique())
    not_signed_up_campaigns = list(set(data[target_column].unique()) - signed_up_campaigns)

    # Prepare the data for prediction
    user_data_repeated = pd.DataFrame([user_data.iloc[0]] * len(not_signed_up_campaigns), columns=user_data.columns)
    user_data_repeated[target_column] = not_signed_up_campaigns

    # Make predictions
    probabilities = model.predict_proba(user_data_repeated[user_columns + [text_column]])[:, 1]
    recommendations = pd.DataFrame({'Campaign': not_signed_up_campaigns, 'Probability': probabilities})
    
    # Return the top N recommendations
    return recommendations.sort_values(by='Probability', ascending=False).head(top_n)

# Example usage
user_id = 'CF1F12A2-046C-E111-B43A-00505680000A'  # Replace with an actual user ID
recommendations = recommend_campaigns_for_user(user_id, model, data, user_columns, text_column, target_column)
print(recommendations)