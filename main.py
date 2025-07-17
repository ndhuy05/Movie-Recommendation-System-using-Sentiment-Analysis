import pandas as pd

extracted_sentiment = pd.read_csv("extracted_sentiments.csv")
rot_10reviews = pd.read_csv('rot_10reviews.csv')
senticnet = pd.read_csv('senticnet.csv')
movie_data = pd.read_csv('data/rotten_tomatoes_movies.csv')
movie_data = movie_data[["id", "title"]]

review_data = pd.read_csv('data/rotten_tomatoes_movie_reviews.csv')
review_data = review_data[["id", "reviewText"]]

movie_review_data = pd.merge(movie_data, review_data, on='id', how='inner')
movie_review_data = movie_review_data.drop('id', axis=1)

columns = ['id', 'title', 'release_date', 'video_release_date',
           'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation',
           'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
           'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
           'Thriller', 'War', 'Western']

movielen = pd.read_csv('ml-100k/u.item', sep='|', names=columns, encoding='latin-1')
movielen = movielen[['title']]
movielen['title'] = movielen['title'].str.replace(r' \(\d{4}\)$', '', regex=True)

merged_data = pd.merge(movie_review_data, movielen, on='title', how='inner')
merged_data = merged_data.dropna()
extracted_sentiment = pd.read_csv('extracted_sentiments.csv')
sentiment_score = pd.read_csv('sentiment_score.csv')