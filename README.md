# movielens-recommender
# MovieLens Recommender System

## Models Built
- **Day 1:** Data Exploration
- **Day 2:** Collaborative Filtering (Neural CF) — MAE: 0.73
- **Day 3:** Content-Based Filtering — MAE: 1.08
  - Features: genres + TF-IDF tags (519 dimensions per movie)
  - Genre weight tuned from 1-5, weight=1 gave best MAE
  - Stronger genre weights produce more coherent recommendations
- **Day 3 (hybrid):** Pending

## Key Insights
- CF learns from user behavior, CB learns from movie content
- CB performs worse on rating prediction but handles cold-start (new movies with no ratings)
- Real systems combine both approaches

## Stack
Python, Pandas, NumPy, Scikit-learn, Keras, Jupyter