from mongoengine import *

class User(Document):
    twitter_id = StringField(unique=True)
    screen_name = StringField(max_length=15, unique=True)
    name = StringField(max_length=100)

class Tweet(Document):
    twitter_id = StringField(unique=True)
    text = StringField(max_length=200, unique=True)
    query = StringField(max_length=100)
    state = StringField(max_length=100)
    user = ReferenceField(User)
