from flask import Flask, render_template, request, jsonify
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
app = Flask(__name__)
stop_words = stopwords.words('english')

def truncated_svd(matrix, k):
    U, Sigma, VT = np.linalg.svd(matrix, full_matrices=False)

    U_k = U[:, :k]  
    Sigma_k = np.diag(Sigma[:k]) 
    VT_k = VT[:k, :] 

    return U_k, Sigma_k, VT_k


# TODO: Fetch dataset, initialize vectorizer and LSA here


newsgroups = fetch_20newsgroups(subset='all')

vectorizer = TfidfVectorizer(max_df=0.95, min_df = 2, max_features=1000, ngram_range=(1, 2), stop_words = stop_words)
tfidf_matrix = vectorizer.fit_transform(newsgroups.data).toarray()

n_components = 100
U_k, Sigma_k, VT_k = truncated_svd(tfidf_matrix, n_components)

lsa_matrix = U_k @ Sigma_k


def search_engine(query):
    """
    Function to search for top 5 similar documents given a query
    Input: query (str)
    Output: documents (list), similarities (list), indices (list)
    """

    query = [query]

    # TODO: Implement search engine here
    # return documents, similarities, indices 

    query_tfidf = vectorizer.transform(query)
    query_lsa = (query_tfidf @ VT_k.T) @ np.linalg.inv(Sigma_k)
    cosine_similarities = cosine_similarity(query_lsa, lsa_matrix)

    print(cosine_similarities)

    top_n = 5
    top_n_indices = cosine_similarities[0].argsort()[::-1][:top_n]

    print(top_n_indices)

    documents = []
    similarities = []
    indicies = top_n_indices.tolist()
    print(type(documents), documents)
    print(type(similarities), similarities)
    print(type(indicies), indicies)

    for index in top_n_indices:
        documents.append(newsgroups.data[index][:2000])
        similarities.append(cosine_similarities[0][index])

    return documents, similarities, indicies

    

    

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    documents, similarities, indices = search_engine(query)
    return jsonify({'documents': documents, 'similarities': similarities, 'indices': indices}) 

if __name__ == '__main__':
    app.run(debug=True)
