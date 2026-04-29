import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.preprocessing import Preprocessor
from model.vectorizer import Vectorizer
from model.naive_bayes import NaiveBayesClassifier
from utils.data_helper import (
    load_csv,
    load_object,
    split_train_valid
)

os.environ["PARRENT_PATH"] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def train(path_to_dataset, train_test_ratio, save_path):
    # create dir
    os.makedirs(save_path, exist_ok=True)

    # Load dataset
    raw_data = load_csv(path_to_dataset)
    # Preprocess dataset
    clean_data = Preprocessor().preprocess_batch(raw_data)

    # Transform text into number(vector)
    vectorizer = Vectorizer(save_path)
    vectorizer.construct_vectorizer(clean_data)

    # Transform label into number representation
    vectors = vectorizer.transform(clean_data['Tweet'])
    labels = vectorizer.encode_label(clean_data['Sentiment'])

    # Split training and validation dataset
    training_resource = split_train_valid(vectors, labels, ratio=train_test_ratio)

    # Train model
    classifier = NaiveBayesClassifier(save_path)
    classifier.train(training_resource)

    # Performance result
    training_result = {
        "train_acc": classifier.training_performance['accuracy'],
        "train_f1": classifier.training_performance['f1'],
        "train_cf_matrix": classifier.training_performance['cf_matrix'],
        "valid_acc": classifier.validation_performance['accuracy'],
        "valid_f1": classifier.validation_performance['f1'],
        "valid_cf_matrix": classifier.validation_performance['cf_matrix'],
        "cross_validation": classifier.training_performance['cross_val']
    }
    return training_result

def predict(text: str, vectorizer: Vectorizer, model: NaiveBayesClassifier):
    preprocessor = Preprocessor()
    text = preprocessor.preprocess(text)
    vectorized_data = vectorizer.transform([text])
    predictions = model.predict(vectorized_data)
    result = {
        "sentiment": vectorizer.decode_label(predictions['sentiment'])[0],
        "negative_probability": round(predictions['probability'][0][0], 4),
        "positive_probability": round(predictions['probability'][0][1], 4)
    }
    return result


def main():
    print(train(
        os.environ.get("PARRENT_PATH")+'/dataset/dataset_tweet_sentiment_selular_service.csv',
        0.2,
        os.environ.get("PARRENT_PATH")+'/model/'))

    vectorizer = load_object(os.environ.get("PARRENT_PATH")+'/model/vectorizer.bin')
    model = load_object(os.environ.get("PARRENT_PATH")+'/model/naive_bayes_model.bin')
    text = "Untung pakai Indihome"
    print(predict(text, vectorizer, model))


if __name__ == "__main__":
    main()
