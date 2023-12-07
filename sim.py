import spacy
import pandas as pd

def calculate_similarity_using_embeddings(keyword_set, text):
    if isinstance(keyword_set, str) and isinstance(text, str):
        nlp = spacy.load("nl_core_news_md")
        keyword_doc = nlp(keyword_set)
        text_doc = nlp(text)
        similarity = keyword_doc.similarity(text_doc)
        return similarity


df = pd.read_csv('final.csv') 



df['Similarity'] = 0.0

for index, row in df.iterrows():
    similarity_score = calculate_similarity_using_embeddings(row['crm_Afspraak_BETREFT_CONTACTFICHE_KeyPhrases'], row['crm_Campagne_Naam_in_email'])
    df.at[index, 'Similarity'] = similarity_score
    print(f"Row {index}: Similarity = {similarity_score}")


df.to_csv('test.csv', index=False)