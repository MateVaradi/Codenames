"""
Obtaining W2V embeddings from various text datasets

dataset used:
https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
https://www.kaggle.com/jrobischon/wikipedia-movie-plots
https://www.kaggle.com/manosss/san-francisco-chronicle-articles-dataset
https://www.kaggle.com/pariza/bbc-news-summary
"""

import pandas as pd
from nltk.tokenize import word_tokenize
from gensim.models import FastText
import time
import re
import os


def clean_document(doc):
    # Make lowercase
    doc = doc.lower()

    # Remove special characters and punctuation
    doc = re.sub('[^a-zA-Z]', ' ', doc)

    # Remove extra whitespaces
    doc = re.sub('\s+', ' ', doc)

    # Tokenization
    doc_list = word_tokenize(doc)

    # Define a (strict list) of stopwords
    stopwords = ['a', 'the', 'and', 'so', 'to', 'as', 'is', 's']
    # Remove stopwords & words shorter than 2 characters long
    doc_list = [w for w in doc_list if (w not in stopwords) and (len(w) > 1)]

    return doc_list

def create_corpus_list_from_df(df, col_name='text'):
    corpus_list = []
    for doc in df[col_name]:
        corpus_list.append(clean_document(doc))

    return corpus_list

def fasttext_train(corpus_list, min_count=10, window=3, size=300, sample=6e-5, alpha=0.03, min_alpha=0.0007,
                   negative=20,
                   workers=1):
    # Initialize model
    model = FastText(min_count=min_count,
                     window=window,
                     size=size,
                     sample=sample,
                     alpha=alpha,
                     min_alpha=min_alpha,
                     negative=negative,
                     workers=workers)

    # Building vocabulary
    model.build_vocab(corpus_list, progress_per=1000)

    # Train model
    t0 = time.time()
    model.train(corpus_list, total_examples=model.corpus_count, epochs=30, report_delay=1)
    print('Time to train the model: {} mins'.format(round((time.time() - t0) / 60, 2)))

    return model


# Load and process data
wiki_data = pd.read_csv('data/wiki_movie_plots.csv')
imdb_data = pd.read_csv('data/IMDB_movie_reviews.csv')
sfc_articles = pd.read_csv('data/sfc_articles.csv')
data = pd.concat([wiki_data['Plot'], imdb_data['review'],
                  sfc_articles['text']],
                 ignore_index=True).to_frame('text')
corpus_list = create_corpus_list_from_df(data)  # process data
# Add BBC news data
news_corpus = []
for folder in os.listdir('data/BBC_articles'):
    for filename in os.listdir(f'data/BBC_articles/{folder}'):
        new_doc = ' '.join(open(f'data/BBC_articles/{folder}/{filename}', 'r').read().split('\n\n'))
        news_corpus.append(clean_document(new_doc))
corpus_list = corpus_list + news_corpus


# Create and save FastText embeddings
ft_model = fasttext_train(corpus_list, window=6, min_count=50, size=200, workers=4)
ft_model.save('embeddings/ft_1')
