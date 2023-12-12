from surprise import Dataset, Reader, SVD
import pandas as pd
from datetime import datetime

# Load your CSV data
df = pd.read_csv('final.csv')
full = df

# Keep only relevant columns
df = df[['crm_Contact_Contactpersoon', 'crm_Campagne_Campagne', 'crm_Inschrijving_Aanwezig_Afwezig', 'crm_Campagne_Naam_in_email', 'crm_Campagne_Startdatum']]
df['crm_Inschrijving_Aanwezig_Afwezig'] = (df['crm_Inschrijving_Aanwezig_Afwezig'] == 'Aanwezig').astype(int)

# Convert 'crm_Campagne_Startdatum' to datetime
df['crm_Campagne_Startdatum'] = pd.to_datetime(df['crm_Campagne_Startdatum'], format='%Y-%m-%d')

# Filter only campaigns in the future
today = pd.Timestamp(datetime.now().date())
df = df[df['crm_Campagne_Startdatum'] > today]

# Create a Surprise Reader and Dataset
reader = Reader(rating_scale=(0, 1))
data = Dataset.load_from_df(df[['crm_Contact_Contactpersoon', 'crm_Campagne_Campagne', 'crm_Inschrijving_Aanwezig_Afwezig']], reader)

# Build the full training set
trainset = data.build_full_trainset()

# Create the SVD model and fit it on the training set
algo = SVD()
algo.fit(trainset)

# Get all unique campaigns
unique_campaigns = df['crm_Campagne_Campagne'].unique()

# Initialize an empty DataFrame to store grouped recommendations
grouped_recommendations = pd.DataFrame(columns=['Campaign', 'Users', 'crm_Campagne_Naam_in_email', 'Likelihood'])

# Iterate over all campaigns and generate recommendations
for campaign in unique_campaigns:
    # Get all users for the current campaign
    users_for_campaign = df[df['crm_Campagne_Campagne'] == campaign]['crm_Contact_Contactpersoon'].unique()
    
    # Create a dataset for the campaign
    full_dataset = [(user, campaign, 0) for user in users_for_campaign]
    
    # Make predictions for the campaign
    campaign_predictions = algo.test(full_dataset)
    
    # Convert predictions to a DataFrame
    pred_df = pd.DataFrame(campaign_predictions, columns=['uid', 'iid', 'r_ui', 'est', 'details'])
    
    # Sort predictions by estimated likelihood in descending order
    pred_df = pred_df.sort_values(by='est', ascending=False)
    
    # Select the top recommendations
    top_recommendations = pred_df[['uid', 'iid', 'est']]
    top_recommendations.columns = ['User', 'Campaign']
    
    # Merge with the original data to get campaign names and details
    top_recommendations = pd.merge(top_recommendations, full, how='left', left_on='Campaign', right_on='crm_Campagne_Campagne')
    top_recommendations = top_recommendations[['User', 'Campaign', 'crm_Campagne_Naam_in_email']]
    
    # Drop duplicates based on the User column
    top_recommendations = top_recommendations.drop_duplicates(subset='User')
    
    # Create a comma-separated string of users
    users_str = ', '.join(top_recommendations['User'])
    
    # Append recommendations to the overall DataFrame
    grouped_recommendations = grouped_recommendations.append({
        'Campaign': campaign,
        'Users': users_str,
        'crm_Campagne_Naam_in_email': top_recommendations.iloc[0]['crm_Campagne_Naam_in_email'],
    }, ignore_index=True)

# Save grouped recommendations to a CSV file
grouped_recommendations.to_csv('grouped_user_recommendations.csv', index=False)

print(f'Grouped Campaign Recommendations saved to grouped_user_recommendations.csv')
