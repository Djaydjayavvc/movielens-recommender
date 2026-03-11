from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split as surprise_split
import json

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

# --- Movies JSON for autocomplete ---
movies_json = movies[['movieId', 'title']].to_dict(orient='records')

# --- Recommend from movie IDs ---
def recommend_from_movies(movie_ids, n=9):
    # Build profile from chosen movies
    valid_ids = [mid for mid in movie_ids if mid in item_features.index]
    if not valid_ids:
        return [], None

    profile = item_features.loc[valid_ids].mean(axis=0).values

    # Cosine similarity
    sims = cosine_similarity([profile], item_features.values)[0]
    sim_series = pd.Series(sims, index=item_features.index)

    # Remove chosen movies
    sim_series = sim_series.drop(index=valid_ids, errors='ignore')

    top_n = sim_series.nlargest(n).reset_index()
    top_n.columns = ['movieId', 'score']
    result = top_n.merge(movies[['movieId', 'title', 'genres']], on='movieId')

    return result.to_dict(orient='records'), profile

def get_wildcard(profile, chosen_ids, n_similar_users=5):
    # Find most similar user to profile
    user_profiles = {}
    for uid in ratings['userId'].unique():
        user_ratings = ratings[ratings['userId'] == uid]
        user_movies = user_ratings['movieId'][user_ratings['movieId'].isin(item_features.index)]
        if len(user_movies) < 5:
            continue
        user_vec = item_features.loc[user_movies].mean(axis=0).values
        user_profiles[uid] = user_vec

    if not user_profiles:
        return None

    user_ids = list(user_profiles.keys())
    user_vecs = np.array(list(user_profiles.values()))
    sims = cosine_similarity([profile], user_vecs)[0]
    most_similar_user = user_ids[np.argmax(sims)]

    # Get their highly rated movies
    user_highly_rated = ratings[
        (ratings['userId'] == most_similar_user) &
        (ratings['rating'] >= 4.0)
    ]['movieId'].values

    # Remove already chosen + already recommended
    candidates = [mid for mid in user_highly_rated
                  if mid not in chosen_ids
                  and mid in item_features.index]

    if not candidates:
        return None

    # Pick the one LEAST similar to the profile (the surprise)
    candidate_features = item_features.loc[candidates]
    sims = cosine_similarity([profile], candidate_features.values)[0]
    least_similar_idx = np.argmin(sims)
    wildcard_id = candidates[least_similar_idx]

    wildcard_movie = movies[movies['movieId'] == wildcard_id].iloc[0]
    return {
        'title': wildcard_movie['title'],
        'genres': wildcard_movie['genres'],
        'similar_user': int(most_similar_user)
    }

# --- Routes ---
@app.route('/')
def index():
    return render_template('index.html', movies_json=json.dumps(movies_json))

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    movie_ids = data.get('movie_ids', [])

    if len(movie_ids) != 3:
        return jsonify({'error': 'Please select exactly 3 movies.'})

    recommendations, profile = recommend_from_movies(movie_ids, n=9)

    if not recommendations:
        return jsonify({'error': 'Could not find those movies in the dataset.'})

    wildcard = get_wildcard(profile, movie_ids)

    return jsonify({
        'recommendations': recommendations,
        'wildcard': wildcard
    })

if __name__ == '__main__':
    app.run(debug=True)