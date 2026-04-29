import os
import pandas as pd


# Query by text search
# Setting variables to be used in format string command below
tweet_count = 5000
text_query = "indihome"
since_date = "2022-01-01"
until_date = "2022-12-31"

# Using OS library to call CLI commands in Python
os.system('snscrape --jsonl --max-results {} --since {} twitter-search "{} until:{}"> ih-2023-data.json'.format(tweet_count, since_date, text_query, until_date))

# Reads the json generated from the CLI command above and creates a pandas dataframe
tweets_df2 = pd.read_json('ih-2023-data.json', lines=True)

# Export dataframe into a CSV
tweets_df2.to_csv('ih-2023-data.csv', sep=',', index=False)

