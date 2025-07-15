import pandas as pd 
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from preprocessing import CleanText

base_path = "../dataset/training.1600000.processed.noemoticon.csv"

#define columns name
cols_name = ["target", "id", "date", "query", "user", "text"]

dataframe = pd.read_csv(base_path, encoding="Latin-1", header=None, names=cols_name, low_memory=False)

#filter positive and negative text, 4 is the binary feeling for positive and 0 is for negative 

dataframe = dataframe[dataframe["target"].isin([0, 4])]
dataframe["sentiment"] = dataframe["target"].map({0:"negative", 4:"Positive"})
dataframe = dataframe.dropna(subset=["text", "sentiment"])

#train the split model
X_train, X_test, y_train, y_test = train_test_split(dataframe["text"], dataframe["sentiment"], test_size=0.25,
                                                    stratify=dataframe["sentiment"], random_state=42)

pipeline = Pipeline([
    ("clean", CleanText()),
    ("tfidf", TfidfVectorizer()),
    ("clf", LinearSVC())
])

#Hyperparameters
param_grid = {
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "tfidf__max_df": [0.9, 0.95],
    "tfidf__min_df": [2, 5],
    "clf__C": [0.5, 1.0, 2.0]
}

#execute gridsearch
grid = GridSearchCV(
    pipeline,
    param_grid=param_grid,
    cv=3,
    n_jobs=3,
    verbose=2
)

#train model
print("Training model...")
grid.fit(X_train, y_train)


best_model = grid.best_estimator_
print("mejores parametros:", grid.best_params_)

#Evaluation
y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred))

#save to .pkl
joblib.dump(best_model, "../models/sentiment_classifier.pkl")
print("Model saved!")


