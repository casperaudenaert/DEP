from surprise import Dataset, Reader, SVD
import pandas as pd
from datetime import datetime
import os

# Your specific date in the format 'YYYY-MM-DD'
specific_date_str = '2023-09-01'
specific_date = datetime.strptime(specific_date_str, '%Y-%m-%d')

df = pd.read_csv('data.csv')
full = df

# Select relevant columns
df = df[['crm_Contact_Contactpersoon', 'crm_Campagne_Campagne', 'crm_Inschrijving_Aanwezig_Afwezig', 'crm_Campagne_Naam_in_email', 'crm_Campagne_Startdatum']]

# Convert 'crm_Campagne_Startdatum' to datetime
df['crm_Campagne_Startdatum'] = pd.to_datetime(df['crm_Campagne_Startdatum'], format='%Y-%m-%d')

# Filter only campaigns in the future based on the specific date
df = df[df['crm_Campagne_Startdatum'] > specific_date]

df['crm_Inschrijving_Aanwezig_Afwezig'] = (df['crm_Inschrijving_Aanwezig_Afwezig'] == 'Aanwezig').astype(int)

reader = Reader(rating_scale=(0, 1))
data = Dataset.load_from_df(df[['crm_Contact_Contactpersoon', 'crm_Campagne_Campagne', 'crm_Inschrijving_Aanwezig_Afwezig']], reader)

trainset = data.build_full_trainset()

algo = SVD()
algo.fit(trainset)

output_folder = 'recommendations'

for user_id in df['crm_Contact_Contactpersoon'].unique():
    user_campaigns = df[df['crm_Contact_Contactpersoon'] == user_id]['crm_Campagne_Campagne'].unique()
    full_dataset = [(user_id, campaign, 0) for campaign in df['crm_Campagne_Campagne'].unique() if campaign not in user_campaigns]

    user_predictions = algo.test(full_dataset)

    pred_df = pd.DataFrame(user_predictions, columns=['uid', 'iid', 'r_ui', 'est', 'details'])

    pred_df = pred_df.sort_values(by='est', ascending=False)

    recommended_campaigns = pred_df[['uid', 'iid', 'est']]
    recommended_campaigns.columns = ['User', 'Campaign', 'Likelihood']
    recommended_campaigns['Likelihood'] *= 100

    # Merge with the full dataset before filtering
    recommended_campaigns = pd.merge(recommended_campaigns, full, how='left', left_on='Campaign', right_on='crm_Campagne_Campagne')

    # Convert 'crm_Campagne_Startdatum' to datetime
    recommended_campaigns['crm_Campagne_Startdatum'] = pd.to_datetime(recommended_campaigns['crm_Campagne_Startdatum'], format='%Y-%m-%d')

    # Filter only campaigns in the future based on the specific date
    recommended_campaigns = recommended_campaigns[recommended_campaigns['crm_Campagne_Startdatum'] > specific_date]

    recommended_campaigns = recommended_campaigns[['User', 'Campaign', 'crm_Campagne_Naam_in_email', 'Likelihood']]
    recommended_campaigns = recommended_campaigns.drop_duplicates(subset='Campaign')

    # Create a folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save the recommendation file only if it is not empty
    if not recommended_campaigns.empty:
        output_file_path = os.path.join(output_folder, f'user_{user_id}_recommendations.csv')
        recommended_campaigns.to_csv(output_file_path, index=False)
        print(f'Recommended Campaigns and Likelihood for User {user_id} saved to {output_file_path}')
    else:
        print(f'No recommendations for User {user_id}')

print('Recommendation process completed.')
