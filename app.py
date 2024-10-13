import matplotlib
matplotlib.use('Agg')  # Use Agg backend for rendering plots without GUI

from flask import Flask, render_template, request
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import io
import base64

# Initialize the Flask app
app = Flask(__name__)

# Fetch the dataset
newsgroups = fetch_20newsgroups(subset='all')

# Create TF-IDF matrix
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(newsgroups.data)

# Apply SVD for LSA (reduce to 100 components)
lsa = TruncatedSVD(n_components=100)
X_reduced = lsa.fit_transform(X)

def process_query(query):
    # Transform query using the same TF-IDF vectorizer and LSA transformation
    query_vec = vectorizer.transform([query])
    query_reduced = lsa.transform(query_vec)
    
    # Compute cosine similarity between query and all documents
    similarities = cosine_similarity(query_reduced, X_reduced).flatten()
    
    # Debugging: print similarities to check
    print(f"Similarities: {similarities}")

    # Get top 5 most similar documents
    top_indices = similarities.argsort()[-5:][::-1]
    top_docs = [(newsgroups.data[i], similarities[i]) for i in top_indices]
    
    return top_docs

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    
    if not query:
        return "No query entered, please go back and enter a query."
    
    top_docs = process_query(query)
    
    # Extract the document text and similarity scores as a list of tuples
    docs_and_scores = [(doc[0], doc[1]) for doc in top_docs]
    
    # Generate a bar chart for similarity scores
    scores = [doc[1] for doc in top_docs]
    img = io.BytesIO()

    plt.clf()  # Clears the current figure
    plt.barh(range(len(scores)), scores, align='center')
    plt.yticks(range(len(scores)), [f"Doc {i+1}" for i in range(len(scores))])
    plt.xlabel('Cosine Similarity')
    plt.tight_layout()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    
    return render_template('results.html', query=query, docs=docs_and_scores, plot_url=plot_url)

if __name__ == '__main__':
    app.run(debug=True)
