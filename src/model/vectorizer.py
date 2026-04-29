import os
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

class Vectorizer:
    def __init__(self, save_path):
        # path = os.environ.get("PARRENT_PATH")+'/model/'
        filename = "/vectorizer.bin"
        self.path = save_path+filename

    def construct_vectorizer(self, dataframe):
        tweet = dataframe['Tweet'].values
        self.vectors = TfidfVectorizer().fit(raw_documents=tweet)
        label = dataframe['Sentiment'].values
        self.labels = LabelEncoder().fit(label)
        self.save_vector()
        return TfidfVectorizer().fit_transform(raw_documents=tweet)

    def save_vector(self):
        with open(self.path, 'wb') as writter:
            pickle.dump(self, writter)

    def transform(self, data):
        X_array = self.vectors.transform(data)
        return X_array

    def encode_label(self, label):
        Y_array = self.labels.transform(label)
        return Y_array

    def decode_label(self, data):
        return self.labels.inverse_transform(data)
