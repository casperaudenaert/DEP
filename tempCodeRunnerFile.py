# df = pd.read_csv('final.csv')
# full = df
# df = df[['crm_Contact_Contactpersoon', 'crm_Campagne_Campagne', 'crm_Inschrijving_Aanwezig_Afwezig','crm_Campagne_Naam_in_email']]

# df['crm_Inschrijving_Aanwezig_Afwezig'] = (df['crm_Inschrijving_Aanwezig_Afwezig'] == 'Aanwezig').astype(int)

# reader = Reader(rating_scale=(0, 1))
# data = Dataset.load_from_df(df[['crm_Contact_Contactpersoon', 'crm_Campagne_Campagne', 'crm_Inschrijving_Aanwezig_Afwezig']], reader)

# trainset = data.build_full_trainset()

# algo = SVD()
# algo.fit(trainset)

# user_id = '01B45481-0877-E911-80FE-001DD8B72B62'

# user_campaigns = df[df['crm_Contact_Contactpersoon'] == user_id]['crm_Campagne_Campagne'].unique()
# full_dataset = [(user_id, campaign, 0) for campaign in df['crm_Campagne_Campagne'].unique() if campaign not in user_campaigns]

# user_predictions = algo.test(full_dataset)

# pred_df = pd.DataFrame(user_predictions, columns=['uid', 'iid', 'r_ui', 'est', 'details'])


# pred_df = pred_df.sort_values(by='est', ascending=False)


# recommended_campaigns = pred_df[['uid', 'iid', 'est']]
# recommended_campaigns.columns = ['User', 'Campaign', 'Likelihood']
# recommended_campaigns['Likelihood'] *= 100

# recommended_campaigns = pd.merge(recommended_campaigns, full, how='left', left_on= 'Campaign', right_on='crm_Campagne_Campagne')
# recommended_campaigns = recommended_campaigns[['User', 'Campaign', 'crm_Campagne_Naam_in_email', 'Likelihood']]
# recommended_campaigns = recommended_campaigns.drop_duplicates(subset='Campaign')


# recommended_campaigns.to_csv('user_recommendations.csv', index=False)

# print(f'Recommended Campaigns and Likelihood for User {user_id} saved to user_recommendations.csv')