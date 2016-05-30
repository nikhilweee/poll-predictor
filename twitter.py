import json
import sys
import autopep8
import requests
import mongoengine
from mongoengine.errors import NotUniqueError
from credentials import *
from requests_oauthlib import OAuth1
from models import *
from data import *

mongoengine.connect(MONGODB_NAME)

auth = OAuth1(client_key=consumer_key,
              client_secret=consumer_secret,
              resource_owner_key=access_token,
              resource_owner_secret=access_token_secret)

def search(state, query):
    state = state.upper()
    placeid = STATES[state]['id']
    params = {
        'q' : query + ' place:' + placeid,
        'count' : 100,
        'lang' : 'en',
    }
    request = requests.get('https://api.twitter.com/1.1/search/tweets.json', auth=auth, params=params)
    return request

def update(request, state, query):
    results = json.loads(request.text)
    if len(results['statuses']) > 0:
        for tweet in results['statuses']:
            u = User(twitter_id=tweet['user']['id_str'], screen_name=tweet['user']['screen_name'], name=tweet['user']['name'])
            try:
                u.save()
            except NotUniqueError:
                pass
            except:
                print(sys.exc_info())
            t = Tweet(twitter_id=tweet['id_str'], text=tweet['text'], query=query, state=state)
            try:
                t.save()
            except NotUniqueError:
                pass
            except:
                print(sys.exc_info())

for candidate in CANDIDATES:
    for state in STATES:
        print(state.lower() + ': ' + candidate)
        request = search(state, candidate)
        print(request.headers.get('x-rate-limit-remaining'))
        update(request, state, candidate)
