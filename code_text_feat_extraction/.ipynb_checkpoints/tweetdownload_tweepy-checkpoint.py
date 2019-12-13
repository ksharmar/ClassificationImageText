#!/usr/bin/env python
# encoding: utf-8
"""A script for downloading (crawling) tweets by their IDs."""

import tweepy
import pickle as pkl

consumerkey = 'zg5ZoWps3wHjfAwjmR5mQHjKW'
consumersecret = 'dsCwQ2Da7ajUTj4IWHTEUY0owJ0GxZOiGVM9Lr0529RqEiRk31'
accesstoken = '936005824111026176-WfU8grlXW2ogKis6vmB1rC6dcAquaCO'
accesssecret = 'nJZBE8h2Y8aloEZnJ4Fuwq3MFJvf6E4BVRx41lwbXqpyc'


if __name__ == '__main__':
    auth = tweepy.OAuthHandler(consumerkey, consumersecret)
    auth.set_access_token(accesstoken, accesssecret)
    api = tweepy.API(auth)
    
    # in_path = './untitled.txt'
    # out_path = './testoutput.pkl'
    
    in_path = '../datasets/waseem/tweet_dataset_ids.txt'
    out_path = '../datasets/waseem/output_status_list.pkl'
    step = 90
    
    all_tweets = []
    tweet_ids = []
    f = open(in_path, 'r') 
    for line in f.readlines():
        id = int(line.strip().split()[0])
        tweet_ids.append(id)
        
    for i in range(0, len(tweet_ids), step):
        tweets = api.statuses_lookup(tweet_ids[i:i+step])
        all_tweets += tweets
        print('done till', i, len(tweets))
        
    pkl.dump(all_tweets, open(out_path, 'wb'))
        
    print('finished')
    
    
