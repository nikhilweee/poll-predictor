import json
import sys
import autopep8
import requests
from credentials import *
from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session

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
