**Supreme Court Petition Classification**


This project classifies Supreme Court certiorari petitions by their Spaeth categorization (a set of 13 issue areas defined in the Spaeth Supreme Court Database). It uses Google Word2Vec embeddings to represent the text of each petition and applies classic tree-based models (e.g., Random Forests, XGBoost) to categorize the petitions.

Key Features
**Input Data:** Full text of certiorari petitions from Supreme Court dockets
**Embedding Model:** Word2Vec embeddings for multidimensional text representation
**Classification:** Tree-based models trained to predict Spaeth categories complete with hyperparameter tuning and cross validation measurement.
