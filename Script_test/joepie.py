import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

data = pd.read_csv('data.csv')
vectorizer = TfidfVectorizer()
x = data.to_string(header=False,
                  index=False,
                  index_names=False).split('\n')
tekst = [','.join(ele.split()) for ele in x]



def vectorize_score(str):
    x = vectorizer.fit_transform([str])
    return x.sum()

data['Similarity'] = 0.0

for row in range(len(tekst)):
    test = tekst[row].replace(',',' ')
    similarity_score = vectorize_score(test)
    data.at[row, 'Similarity'] = similarity_score
    print(f"Row {row}: Similarity = {similarity_score}")

data.to_csv('test.csv', index=False)


# for index, row in data.iterrows():
#     print(tf.fit(row))


