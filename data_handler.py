import json
import pdb
import codecs
import pdb
import pandas as pd
import os
import numpy as np

#dir_path = os.path.dirname(os.path.realpath(__file__))

encoded_data_path = "encoded_data/"

def get_data():
    tweets = []
    #files = ['racism.json', 'neither.json', 'sexism.json']
    files = ['racism.json', 'sexism.json', 'amateur_expert.json']
    for file in files:
        with open('dataset/' + file, 'r', encoding='utf-8') as f:
            data = f.readlines()
        
        for line in data:
            tweet_full = json.loads(line)
            tweets.append({
                'id': tweet_full['id'],
                'text': tweet_full['text'].lower(),
                'label': tweet_full['Annotation'].lower(),
                'name': tweet_full['user']['name'].split()[0].lower()
                })

    return tweets


def get_data_dataframe(data):
    return pd.DataFrame.from_dict(data)



tweets = get_data()
tweets_df = get_data_dataframe(tweets)

# saving the dataframe
tweets_df.to_pickle(encoded_data_path + "tweets.pkl")

males, females = {}, {}

with open('dataset/' + 'males.txt', 'r', encoding='utf-8') as f:
    males = set([w.strip().lower() for w in f.readlines()])
    
with open('dataset/' + 'females.txt', 'r',  encoding='utf-8') as f:
    females = set([w.strip().lower() for w in f.readlines()])

males_c, females_c, not_found = 0, 0, 0

for t in tweets:
    if t['name'] in males:
        males_c += 1
    elif t['name'] in females:
        females_c += 1
    else:
        not_found += 1
print(males_c, females_c, not_found) #714 535 11066
    
    
