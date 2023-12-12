from flask import Flask, render_template, request, send_file
import pandas as pd
from surprise import Dataset, Reader, SVD, KNNBasic

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommendations', methods=['GET', 'POST'])
def recommendations():
    if request.method == 'POST':
        user_id = request.form['user_id']
        
        df = pd.read_csv('test.csv')
        full = df
        df = df[['crm_Contact_Contactpersoon', 'crm_Campagne_Campagne', 'Similarity','crm_Campagne_Naam_in_email']]


        reader = Reader(rating_scale=(0, 10))
        data = Dataset.load_from_df(df[['crm_Contact_Contactpersoon', 'crm_Campagne_Campagne', 'Similarity']], reader)

        trainset = data.build_full_trainset()

        algo = KNNBasic(sim_options={
            "name": "cosine",
            "user_based":False,
        })
        algo.fit(trainset)

        user_id = '01B45481-0877-E911-80FE-001DD8B72B62'

        user_campaigns = df[df['crm_Contact_Contactpersoon'] == user_id]['crm_Campagne_Campagne'].unique()
        full_dataset = [(user_id, campaign, 0) for campaign in df['crm_Campagne_Campagne'].unique() if campaign not in user_campaigns]

        user_predictions = algo.test(full_dataset)
        print(user_predictions)

        pred_df = pd.DataFrame(user_predictions, columns=['uid', 'iid', 'r_ui', 'est', 'details'])
        pred_df = pred_df.sort_values(by='est', ascending=False)


        recommended_campaigns = pred_df[['uid', 'iid', 'est']]
        recommended_campaigns.columns = ['User', 'Campaign', 'Likelihood']

        recommended_campaigns = pd.merge(recommended_campaigns, full, how='left', left_on= 'Campaign', right_on='crm_Campagne_Campagne')
        recommended_campaigns = recommended_campaigns[['User', 'Campaign', 'crm_Campagne_Naam_in_email', 'Likelihood']]
        recommended_campaigns = recommended_campaigns.drop_duplicates(subset='Campaign')

        file_path = f'user_recommendations_{user_id}.csv'
        recommended_campaigns.to_csv(file_path, index=False)
        
        return render_template('recommendations.html', user_id=user_id, recommendations=recommended_campaigns, file_path=file_path)
    
    return render_template('recommendations.html')

@app.route('/download_recommendations/<filename>')
def download_recommendations(filename):
    return send_file(filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
