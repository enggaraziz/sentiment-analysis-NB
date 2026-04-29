import json
import os
import tweepy
import pandas as pd
import pickle
from datetime import datetime

from src.wrapper import predict


def read_performance_data(path:str):
    data = {}
    for dtype in ['validation', 'train']:
        with open(path+f'{dtype}_metrics_score.json', 'r') as reader:
            data[dtype] = json.load(reader)
    return data


def _write_tweet_doc_log(log_data):
    df = pd.DataFrame.from_records([log_data])
    df.to_csv(os.environ['TWEETS_LOG_PATH'], mode='a', index=False, header=False)


def get_tweet_data(model_name, keyword, filters, result_type, total_data=25, exclude_filter=False):
    cur_date = datetime.now()
    filter_mode = '-filter:' if exclude_filter else 'filter'
    q_filter = []
    for item in filters:
        q_filter.append(filter_mode+item)
    
    query = '{} {}'.format(
        keyword,
        ' '.join(q_filter)
    )

    auth = tweepy.OAuth1UserHandler(
        os.environ["API_KEY"],
        os.environ["API_KEY_SECRET"],
        os.environ["ACCESS_TOKEN"],
        os.environ["ACCESS_TOKEN_SECRET"],
    )

    api = tweepy.API(auth)
    tweets_result = api.search_tweets(
        q=query,
        count=int(total_data),
        result_type=result_type
    )

    tweets = []
    for tweet in tweets_result:
        d = {}
        d['username'] = '@'+str(tweet.user.screen_name)
        d['created_at'] = str(tweet.created_at)
        d['tweet'] = tweet.text.replace('\n', ' ')
        tweets.append(d)
    if len(tweets):
        # do something here
        print("[TWEET-SCRAPER] Found {} tweets. Begin analyzing tweet".format(len(tweets)))
        doc_name_id = '{}_{}'.format(
            str(cur_date.year+cur_date.month+cur_date.day),
            str(cur_date.hour+cur_date.minute+cur_date.second)
        )
        doc_name = 'tweet_{}_{}_{}.csv'.format(
            keyword,
            model_name,
            doc_name_id
        )
        doc_path = os.environ['TWEETS_DOCS_PATH']+doc_name

        # construct tweet df
        df = pd.DataFrame.from_records(tweets)

        # keywords,filters,filter_exclude_mode,created_at,model_analyzer,filename,status
        log_data = {
            'keywords': keyword,
            'filters': ', '.join(filters),
            'filter_exclude_mode': 'On' if exclude_filter else 'Off',
            'created_at': cur_date.strftime('%Y-%m-%d %H:%M:%S'),
            'model_analyzer': model_name,
            'filename': doc_name,
            'total_tweet': len(tweets),
            'status': "Finished"
        }
        # write log
        _write_tweet_doc_log(log_data)

        # analyze tweet, add sentiment and probability column
        # TODO: 
        df = _analyze_tweet(model_name, df)
        # save
        df.to_csv(doc_path, mode='a', header=True, index=False)
        return True 
    else:
        print("[TWEET-SCRAPER] Found {} tweets. Analyzing tweet failed".format(len(tweets)))
        return False

def __load_pickle_object(path):
    with open(path, 'rb') as reader:
        return pickle.load(reader)

def _analyze_tweet(model_name, tweet_df):
    model_file = '/naive_bayes_model.bin'
    vectorizer_file = '/vectorizer.bin'
    base_path = os.environ['MODEL_PATH'] + model_name

    model = __load_pickle_object(base_path + model_file)
    vectorizer= __load_pickle_object(base_path + vectorizer_file)

    tweet_df = __predict_sentiment(model, vectorizer, tweet_df)
    return tweet_df


def __predict_sentiment(model, vectorizer, tweet_df):
    tweets = tweet_df['tweet'].tolist()
    sentiment = []
    confidence = []
    for tweet in tweets:
        result = predict(tweet, vectorizer, model)
        sentiment.append(result['sentiment'])
        confidence.append(result['{}_probability'.format(result['sentiment'])])

    tweet_df['sentiment'] = sentiment
    tweet_df['confidence_probability'] = confidence
    return tweet_df

def check_model_exist(model_log_path):
    # check model csv
    df = pd.read_csv(model_log_path)
    len_data = len(df)
    if len_data:
        model_list = df["save_path"].tolist()
        model_list = [save_path.split('/')[-1] for save_path in model_list]
        return model_list
    return []
    
def check_tweet_doc_exist(tweet_doc_path):
    # check model csv
    df = pd.read_csv(tweet_doc_path)
    len_data = len(df)
    if len_data:
        tweet_list = df["filename"].tolist()
        tweet_list = [filename.split('/')[-1] for filename in tweet_list]
        return tweet_list
    return []
    
def read_tweet_analysis_data(path:str):
    # provide following data
    # 1. Line chart: need grouped tweet each day , both positive, and negative 
    # 2. Text : detect min and max date
    # 3. Text : detect total pos and neg sentiment
    # 4. Text : Average model probability value on each class
    # 5. Text : Total Data
    data = {}
    tweet_df = pd.read_csv(path)
    print(tweet_df)
    tweet_df['created_at'] = pd.to_datetime(tweet_df['created_at'])
    df_group = tweet_df.groupby([tweet_df['created_at'].dt.date])

    # grouped_sentiment
    sentiment_count = {
        'positive' : [],
        'negative' : []
    }
    date_list = []

    for group in df_group:
        pos_count = 0
        neg_count = 0
        sentiments = group[1]['sentiment'].tolist()
        for sentiment in sentiments:
            if sentiment == 'positive':
                pos_count = pos_count + 1
            else:
                neg_count = neg_count + 1
        sentiment_count['positive'].append(pos_count)
        sentiment_count['negative'].append(neg_count)
        date_list.append(str(group[0]))

    # data 
    data['dates'] = date_list
    data['max_date'] = max(date_list)
    data['min_date'] = min(date_list)
    data['total_data'] = len(tweet_df)
    data['average_prob_neg'] = 100*round(tweet_df.loc[tweet_df['sentiment'] == 'negative'].loc[:, 'confidence_probability'].mean(), 2)
    data['average_prob_pos'] = 100*round(tweet_df.loc[tweet_df['sentiment'] == 'positive'].loc[:, 'confidence_probability'].mean(), 2)
    data['total_neg'] = sum(sentiment_count['negative'])
    data['total_pos'] = sum(sentiment_count['positive'])
    data['tweet_series'] = sentiment_count
    print(data)
    return data, tweet_df