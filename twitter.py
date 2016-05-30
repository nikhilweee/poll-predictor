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
from credentials import *
from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session

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


client = BackendApplicationClient(client_id=consumer_key)
print("New Session . . .")
twitter = OAuth2Session(client=client)
print("Fetching token . . .")
twitter.fetch_token(token_url='https://api.twitter.com/oauth2/token',
                    client_id=consumer_key,
                    client_secret=consumer_secret)
query = ' '.join(sys.argv[1:])
payload = {
    'q': query,
    'count' : 100,
    'lang' : 'en',
}

print('Requesting Tweets . . .')
r = twitter.request('GET', 'https://api.twitter.com/1.1/search/tweets.json', params=payload)
pythontext = json.loads(r.text)
tweetarray = set()
for tweet in pythontext['statuses']:
    tweetarray.add(tweet['text'])
filename = query.replace(' ', '-')
filename = filename + '.py'
print filename
f = open(filename, 'w')
tweetarray = autopep8.fix_code(str(tweetarray), options={'aggressive' : 1})
f.write(tweetarray)
f.close()

