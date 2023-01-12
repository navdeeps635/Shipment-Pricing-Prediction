import pymongo
import os

client = pymongo.MongoClient(os.getenv("MONGO_DB_URL")) 
db = client.test

print('connection OK')

