user_id_to_recommend = '25819DD0-28CE-E811-80F7-001DD8B72B61'
def find_best_n_components(data, n_components_list, vectorizer_params_list, scaler_params_list):
    best_score = 0
    best_n_components = 0
    best_vectorizer_params = None
    best_scaler_params = None

    for n_components in n_components_list:
        for vectorizer_params in vectorizer_params_list:
            for scaler_params in scaler_params_list:
                cv_truncated = TruncatedSVD(n_components=n_components)
                user_feature_matrix_truncated = cv_truncated.fit_transform(TfidfVectorizer(**vectorizer_params).fit_transform(data['user_features']))

                scaler = StandardScaler(**scaler_params)
                user_feature_matrix_normalized = scaler.fit_transform(user_feature_matrix_truncated)

                user_similarity = cosine_similarity(user_feature_matrix_normalized)

                attended_campaigns = data[data['Contactfiche'] == user_id_to_recommend].index.tolist()
                simulated_predictions = np.mean(user_similarity[attended_campaigns], axis=0)
                true_labels = np.zeros(len(data))
                true_labels[attended_campaigns] = 1
                binary_predictions = (simulated_predictions >= 0.5).astype(int)

                precision = precision_score(true_labels, binary_predictions)
                recall = recall_score(true_labels, binary_predictions)
                f1 = f1_score(true_labels, binary_predictions)

                # You can add more metrics here if needed

                # You may want to use a weighted sum or other aggregation method based on your priorities
                aggregated_score = (precision + recall + f1) / 3  # Adjust this based on your priorities

                # Update the best parameters if the current configuration gives a higher aggregated score
                if aggregated_score > best_score:
                    best_score = aggregated_score
                    best_n_components = n_components
                    best_vectorizer_params = vectorizer_params
                    best_scaler_params = scaler_params
                print(f'Score: {aggregated_score} with n_components={best_n_components}, vectorizer_params={best_vectorizer_params}, and scaler_params={best_scaler_params}')

    return best_n_components, best_vectorizer_params, best_scaler_params

# Define a range of n_components values to search over
n_components_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

# Define a range of vectorizer parameters to search over
vectorizer_params_list = [
    {'max_features': 5000, 'ngram_range': (1, 1)},
    {'max_features': 10000, 'ngram_range': (1, 1)},
    {'max_features': None, 'ngram_range': (1, 1)},
    {'max_features': 5000, 'ngram_range': (1, 2)},
    {'max_features': 10000, 'ngram_range': (1, 2)},
    {'max_features': None, 'ngram_range': (1, 2)},
]

# Define a range of scaler parameters to search over
scaler_params_list = [{'with_mean': True, 'with_std': True}, {'with_mean': False, 'with_std': True}]

# Perform the grid search
#best_n_components, best_vectorizer_params, best_scaler_params = find_best_n_components(df, n_components_list, vectorizer_params_list, scaler_params_list)

best_n_components = 70
best_vectorizer_params = {'max_features': 10000, 'ngram_range': (1, 1)}
best_scaler_params= {'with_mean': True, 'with_std': True}
# Print the best parameters
print(f"Best n_components: {best_n_components}")
print(f"Best vectorizer parameters: {best_vectorizer_params}")
print(f"Best scaler parameters: {best_scaler_params}")