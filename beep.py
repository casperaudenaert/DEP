from surprise import Dataset, Reader
from surprise.model_selection import cross_validate
from surprise import SVD, KNNBasic, NMF
from surprise.accuracy import rmse
from surprise.model_selection import train_test_split

# Load your dataset
data = pd.read_csv('data.csv')

# Assuming each row in your dataset is a user-campaign interaction
# Convert the dataset into a format suitable for Surprise
user_id_column = 'crm_Contact_Contactpersoon'
campaign_id_column = 'crm_Campagne_Campagne'

# Create a dataframe with user, item, and ratings (interactions)
# For this example, let's assume a binary interaction (1 for interaction)
interaction_data = data[[user_id_column, campaign_id_column]]
interaction_data['interaction'] = 1  # or use some interaction count if available

# Load the dataset into Surprise
reader = Reader(rating_scale=(0, 1))
data = Dataset.load_from_df(interaction_data[[user_id_column, campaign_id_column, 'interaction']], reader)

# Define a cross-validation iterator (e.g., 5-fold cross-validation)
from surprise.model_selection import KFold
kf = KFold(n_splits=5)

# Define and test multiple algorithms
for algorithm in [SVD(), KNNBasic(), NMF()]:
    print(f'\nEvaluating {algorithm.__class__.__name__} algorithm...')
    cross_validate(algorithm, data, measures=['RMSE', 'MAE'], cv=kf, verbose=True)
