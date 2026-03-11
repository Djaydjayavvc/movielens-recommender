from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split as surprise_split

app = Flask(__name__)

# --- Load data ---
ratings = pd.read_csv('data/ratings.csv')
movies  = pd.read_csv('data/movies.csv')
tags    = pd.read_csv('data/tags.csv')

# --- Build genre matrix ---
movies['genre_list'] = movies['genres'].str.split('|')
mlb = MultiLabelBinarizer()
genre_matrix = pd.DataFrame(
    mlb.fit_transform(movies['genre_list']),
    index=movies['movieId'],
    columns=mlb.classes_
).drop(columns=['(no genres listed)'], errors='ignore')

# --- Build tag matrix ---
tag_docs = (
    tags.groupby('movieId')['tag']
    .apply(lambda x: ' '.join(x.astype(str).str.lower()))
    .reset_index()
    .rename(columns={'tag': 'tag_doc'})
)
movies_with_tags = movies[['movieId']].merge(tag_docs, on='movieId', how='left')
movies_with_tags['tag_doc'] = movies_with_tags['tag_doc'].fillna('')
tfidf = TfidfVectorizer(max_features=500, stop_words='english')
tag_matrix = pd.DataFrame(
    tfidf.fit_transform(movies_with_tags['tag_doc']).toarray(),
    index=movies_with_tags['movieId']
)

# --- Combine features ---
item_features = pd.concat([genre_matrix, tag_matrix], axis=1).fillna(0)

# --- Train SVD ---
reader = Reader(rating_scale=(0.5, 5))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
trainset, _ = surprise_split(data, test_size=0.2, random_state=42)
svd_model = SVD(n_factors=10, random_state=42)
svd_model.fit(trainset)

# --- User profile function ---
def build_user_profile(user_id, ratings_df, item_features_df, decay=0.5):
    user_ratings = ratings_df[ratings_df['userId'] == user_id].copy()
    user_ratings['rating_centered'] = user_ratings['rating'] - user_ratings['rating'].mean()
    max_timestamp = ratings_df['timestamp'].max()
    seconds_per_year = 365.25 * 24 * 60 * 60
    user_ratings['years_ago'] = (max_timestamp - user_ratings['timestamp']) / seconds_per_year
    user_ratings['recency_weight'] = np.exp(-decay * user_ratings['years_ago'])
    user_ratings['final_weight'] = user_ratings['rating_centered'] * user_ratings['recency_weight']
    user_ratings = user_ratings[user_ratings['movieId'].isin(item_features_df.index)]
    if user_ratings.empty:
        return None
    rated_features = item_features_df.loc[user_ratings['movieId']]
    weights = user_ratings['final_weight'].values
    return np.dot(weights, rated_features.values)

# --- Hybrid recommend function ---
def hybrid_recommend(user_id, n=10, alpha=0.2):
    profile = build_user_profile(user_id, ratings, item_features)
    if profile is None:
        return []
    sims = cosine_similarity([profile], item_features.values)[0]
    cb_scores = pd.Series(sims, index=item_features.index)
    all_movies = item_features.index.tolist()
    cf_scores = pd.Series(
        [svd_model.predict(user_id, mid).est for mid in all_movies],
        index=all_movies
    )
    cb_norm = (cb_scores - cb_scores.min()) / (cb_scores.max() - cb_scores.min())
    cf_norm = (cf_scores - cf_scores.min()) / (cf_scores.max() - cf_scores.min())
    hybrid_scores = alpha * cb_norm + (1 - alpha) * cf_norm
    already_rated = ratings[ratings['userId'] == user_id]['movieId'].values
    hybrid_scores = hybrid_scores.drop(index=already_rated, errors='ignore')
    top_n = hybrid_scores.nlargest(n).reset_index()
    top_n.columns = ['movieId', 'score']
    result = top_n.merge(movies[['movieId', 'title', 'genres']], on='movieId')
    return result.to_dict(orient='records')

# --- Routes ---
@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = []
    user_id = None
    error = None
    if request.method == 'POST':
        try:
            user_id = int(request.form['user_id'])
            if user_id not in ratings['userId'].values:
                error = f"User {user_id} not found. Try a number between 1 and 610."
            else:
                recommendations = hybrid_recommend(user_id)
        except ValueError:
            error = "Please enter a valid number."
    return render_template('index.html',
                         recommendations=recommendations,
                         user_id=user_id,
                         error=error)

if __name__ == '__main__':
    app.run(debug=True)