import pandas as pd
import numpy as np

# Random Forest Modeling
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split
from scipy.stats import randint, uniform
import random
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

# XG Boost
import xgboost as xgb

# Ensemble Model
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier

# Cross Validation
from sklearn.model_selection import StratifiedKFold
# Use stratified since some folds don't have any instances of a certain class.

# Tf - Idf Calculation
from sklearn.feature_extraction.text import TfidfVectorizer

# Accuracy Measures
from sklearn.metrics import accuracy_score, precision_score, f1_score
from gensim.models import KeyedVectors



word2vec_model = KeyedVectors.load_word2vec_format('C:/Work/Embedding/GoogleNews-vectors-negative300.bin/GoogleNews-vectors-negative300.bin', binary=True)

# For a given document creates a dictionary with each word in document mapped to its tf-idf score
def create_tfidf_dict(df):
    # Compute tf idf values
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['Text']) 
    #Rows represent documents (writs)
    #Columns represent terms (unique words in a document).
    tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()
    # Returns a list of our matrix (the unique words)

    tfidf_scores = {}
    for index, document in enumerate(df['Text']): # loop through every document
        tfidf_scores[index] = {}
        for word in document.split(): # Get every word in that document
            if word in tfidf_vectorizer.vocabulary_: #If we have the tf idf for that word
                tfidf_scores[index][word] = tfidf_matrix[index, tfidf_vectorizer.vocabulary_[word]] #tfidf_vectorizer returns column index of a word in our tfidf matrix
    return tfidf_scores

# For each document returns an 'average' vector that represents an 'average' of all the words in document
# Use 'want_tf_idf' if you wanna weight each vector by its tf-idf score
# Otherwise, if you set want_tf_idf to False, it returns a simple arithmetic average of each vector in the document (component by component)
def document_to_embedding(doc, word2vec_model, want_tf_idf, doc_index=None, tf_idf_dict=None):
    if not want_tf_idf:
        words = doc.split()  # Tokenize the document
        embeddings = []

        for word in words:
            if word in word2vec_model:
                embeddings.append(word2vec_model[word])

        if len(embeddings) == 0:  # If no word has embedding, return zero vector
            return np.zeros(word2vec_model.vector_size)
        
        # Return the average embedding for the document
        return np.mean(embeddings, axis=0)
    else:
        words = doc.split()  # Tokenize the document
        embeddings = []
        total_weight = 0  # To keep track of the total tfidf weight to divide by at the end (normalize)
        
        for word in words:
            if word in word2vec_model and word in tf_idf_dict[doc_index]:
                weight = tf_idf_dict[doc_index][word]  # Get the TF-IDF score of the word
                embeddings.append(word2vec_model[word] * weight)
                total_weight += weight
        
        if len(embeddings) == 0 or total_weight == 0:  # If no word has embedding, return zero vector
            return np.zeros(word2vec_model.vector_size)
    
        # Return the weighted average embedding for the document
        return np.sum(embeddings, axis=0) / total_weight


def prepareData(df):
    # Create a list of our 300-dimensional 'average' vector of a document
    vector_df = pd.DataFrame(df['Vector'].tolist(), index=df.index)
    X = vector_df
    Y = df['issueArea']
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)
    return X_train, X_test, y_train, y_test

def printAccuracy(best_rf, best_params, X_train, y_train, X_test, y_test):
    # Cross-validation on training set only
    cv_scores = cross_val_score(best_rf, X_train, y_train, cv=5)
    y_pred_train = cross_val_predict(best_rf, X_train, y_train, cv=5)
    conf_matrix_train = confusion_matrix(y_train, y_pred_train)

    # Print training set cross-validation results
    print('Best Model:', best_rf)
    print('Best Hyperparameters:', best_params)
    print("Cross-validated accuracy scores on training set:", cv_scores)
    print("Average cross-validated accuracy on training set:", np.mean(cv_scores))
    print("Confusion matrix on training set:\n", conf_matrix_train)

    # Predictions and metrics on test set
    y_pred_test = best_rf.predict(X_test)
    accuracy_test = accuracy_score(y_test, y_pred_test)
    precision_test = precision_score(y_test, y_pred_test, average='weighted')
    f1_test = f1_score(y_test, y_pred_test, average='weighted')

    print(f"Test set accuracy: {accuracy_test:.3f}")
    print(f"Test set precision: {precision_test:.3f}")
    print(f"Test set F1 score: {f1_test:.3f}")

def runRandomForest(df):
    X_train, X_test, y_train, y_test = prepareData(df)

    rf = RandomForestClassifier()
    param_dist = {
        'n_estimators': randint(100, 400),
        'max_features': ['auto', 'sqrt'],
        'max_depth': randint(3, 25),
        'min_samples_split': randint(2, 11),
        'min_samples_leaf': randint(1, 11),
        'bootstrap': [True, False]
    }

    rand_search = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=20, cv=5)
    rand_search.fit(X_train, y_train)

    # Best model and parameters
    best_rf = rand_search.best_estimator_
    best_params = rand_search.best_params_

    # Print results
    printAccuracy(best_rf, best_params, X_train, y_train, X_test, y_test)

def runXG(df):
    X_train, X_test, y_train, y_test = prepareData(df)

    xg = xgb.XGBClassifier(
        use_label_encoder=False,
        eval_metric='mlogloss',  # Use multiclass log loss
        objective='multi:softmax'  # Set to multi:softmax for discrete class predictions
    )

    param_dist2 = {
        'max_depth': randint(1, 7),
        'min_child_weight': randint(1, 5),
        'learning_rate': uniform(0.0001, 0.4),  # Continuous distribution from 0.0001 to 0.4
        'n_estimators': randint(50, 250),  # Random integer between 50 and 250
        'subsample': uniform(0.5, 1.0)  # Uniform distribution from 0.5 to 1.0
    }

    #stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True)
    rand_search = RandomizedSearchCV(xg, param_distributions=param_dist2, n_iter=20, cv = 5)
    # IMPORTANT, n_iter is how many random combos of hyperparameters the rand_search does. The higher the better, but it will take longer.
    rand_search.fit(X_train, y_train)

    # Best model and parameters
    best_rf = rand_search.best_estimator_
    best_params = rand_search.best_params_

    # Print results
    printAccuracy(best_rf, best_params, X_train, y_train, X_test, y_test)

def main():
    #og_df = pd.read_parquet('C:/Work/MachineLearning/Files/doj_writs_spaeth.parquet')
    og_df = pd.read_parquet("C:/Work/Embedding/Files/combined_petitions_spaeth_and_full_text.parquet")
    #df = og_df[~og_df['SPAETH Issue'].str.contains('Not Found')].reset_index(drop=True)
    df = og_df[~og_df['issueArea'].str.contains('Not Found')].reset_index(drop=True)

    # Remove troublesome classes below
    # Class 11 DNE and smaller classes mess up cross validation as their total 
    # number of observations is less than the number of folds in cross validation
    df = df[df['issueArea'] != 11]
    df = df.groupby('issueArea').filter(lambda x: len(x) > 2)
    df['issueArea'] = df['issueArea'].astype('category').cat.codes

    df['Text'] = df['Text'].apply(lambda x: ' '.join(x.split())) # Remove extra trailing whitespace
    tf_idf_dict = create_tfidf_dict(df)
    #Create the embedded vector for each document (writ)
    df['Vector'] = df.apply(lambda row: document_to_embedding(row['Text'], word2vec_model, False, row.name, tf_idf_dict), axis=1)
    runRandomForest(df)
    runXG(df)
    

if __name__ == "__main__":
    main()



