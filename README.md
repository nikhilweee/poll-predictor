# Poll Predictor
An app to predict the results of the US Presidential Elections of 2016.
Based on Sentiment Analysis using CNNs.

## First Steps
The `twitter.py` script returns upto 100 tweets using the twitter API, but Twitter requires API requests to be authenticated using OAuth2.

To do so, first obtain your credentials by creating an app on [apps.twitter.com](https://apps.twitter.com/) to get a `consumer_key` and `consumer_secret`

Then, create a file named `credentials.py` and key in the following
```python
consumer_key = '<your_consumer_key>'
consumer_secret = '<your_consumer_secret>''
```

Also, make sure to install the dependencies using
```
$ pip install -r requirements.txt
```

## Usage
Just run the script with the search query as arguments. For example, to search for *Barack Obama*, just run
```
$ python twitter.py barack obama
```
This should generate a file named `barack-obama.py` with the tweets listed.
