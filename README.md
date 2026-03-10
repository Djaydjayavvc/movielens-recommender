# MovieLens Recommender System

A movie recommendation system built on the MovieLens 100K dataset, 
implementing and comparing multiple approaches from collaborative 
filtering to content-based filtering and a hybrid model.

## Models & Results

| Model | MAE | Notes |
|-------|-----|-------|
| SVD (Collaborative Filtering) | 0.6734 | Best raw accuracy |
| Neural CF (Embeddings + Dropout) | 0.7300 | Deep learning approach |
| Hybrid (CB + SVD, alpha=0.2) | 0.8145 | Best cold-start coverage |
| Content-Based Filtering | 1.0749 | Explainable, handles new movies |

## Project Structure
```
movielens-recommender/
├── data/               # MovieLens 100K dataset
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_collaborative_filtering.ipynb
│   └── 03_content_based_filtering.ipynb
├── src/                # (future) reusable modules
└── README.md
```

## Notebooks

**01 - Data Exploration**
- Dataset overview, rating distributions, user and movie statistics

**02 - Collaborative Filtering**
- Neural network with user/movie embeddings (size=50)
- Dropout regularization (0.3) to prevent overfitting
- MAE: 0.7300

**03 - Content-Based Filtering & Hybrid**
- Movie features: genres (binary) + TF-IDF tags (500 features)
- User profiles with exponential recency decay (decay=0.5)
- SVD collaborative filtering (n_factors=10)
- Hybrid blend of CB + SVD (alpha=0.2)
- Full hyperparameter search across all parameters

## Hyperparameter Tuning

| Parameter | Values Tested | Best | MAE |
|-----------|--------------|------|-----|
| genre_weight | 1,2,3,4,5 | 1 | 1.0805 |
| max_features (TF-IDF) | 100,200,500,1000 | 500 | 1.0805 |
| recency decay | 0,0.05,0.1,0.2,0.5,0.7,1.0,1.5,2.0 | 0.5 | 1.0749 |
| svd_n_factors | 10,25,50,100,200 | 10 | 0.6734 |
| hybrid alpha | 0.0,0.2,0.4,0.5,0.6,0.8,1.0 | 0.2 | 0.8145 |

## Key Insights

- SVD outperforms Neural CF on this dataset size — simpler models 
  win when data is limited
- Content-based has higher MAE but handles cold-start (new movies 
  with no ratings) which CF cannot
- Recency decay improves CB performance — recent ratings are more 
  representative of current taste
- Hybrid trades accuracy for coverage — useful in production where 
  cold-start is a real problem

## Stack
Python, Pandas, NumPy, Scikit-learn, Keras, Scikit-surprise, Jupyter

## Future Improvements
- Add TMDB metadata (director, cast) as content features
- Time-based train/test split for more realistic evaluation
- Precision@K and Recall@K metrics for recommendation quality
- Flask web interface for live demo
