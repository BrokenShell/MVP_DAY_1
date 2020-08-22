"""
NLP Model for DS Build Week

Input -> TF-IDF -> KNN -> Output
"""
import pandas as pd
from os import getenv
from pymongo import MongoClient
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer


class PredictionBot:
    """ NLP bot for Cannabis Suggestion App """
    db = MongoClient(
        f"mongodb+srv://{getenv('MONGODB_USER')}:{getenv('MONGODB_PASS')}"
        f"@{getenv('MONGODB_URI')}/test?retryWrites=true&w=majority"
    ).medcabin.strains
    df = pd.read_csv('data/cannabis.csv')
    tfidf = TfidfVectorizer(ngram_range=(1, 3), max_features=8000)
    knn = NearestNeighbors(n_neighbors=1, n_jobs=-1)
    tokens = tfidf.fit_transform(
        df['Description'] + ' ' + df['Effects'] + ' ' + df['Flavors']
    )
    nearest = knn.fit(
        pd.DataFrame(tokens.todense(), columns=tfidf.get_feature_names())
    )

    def predict(self, user_input) -> dict:
        return next(self.db.find({'_id': int(self.nearest.kneighbors(
            self.tfidf.transform([user_input]).todense()
        )[1][0][0])}))


if __name__ == '__main__':
    bot = PredictionBot()
    print(bot.predict("I don't want psychoactive effects"))
