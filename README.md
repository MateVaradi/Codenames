# Codenames bot
A simple algorithm based on Word2Vec embeddings to automatically generate clues for the board game [Codenames](https://en.wikipedia.org/wiki/Codenames_(board_game)).  

## Getting started

### Prerequisites
Apart from standard Python data science libraries you will need to have `nltk` and `gensim` installed.

### How to run

1. Download the following datasets and place them in the data folder of this repository:
+ https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
+ https://www.kaggle.com/jrobischon/wikipedia-movie-plots
+ https://www.kaggle.com/manosss/san-francisco-chronicle-articles-dataset
+ https://www.kaggle.com/pariza/bbc-news-summary
2. Run `create_embeddings.py`. This will take some time to run.
3. Run the examples from `create_clues.py` to see how the clue generation works and start creating clues for your own Codenames board. 
 
[Codenames board]: https://github.com/MateVaradi/Codenames/blob/main/illustrations/SetUp.jpg "Codenames board example"

