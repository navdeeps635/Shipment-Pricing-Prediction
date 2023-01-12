import pymongo
import json
import os
from dataclasses import dataclass
  
    #provide the mongodb url to connect python to mongodb.
    mongodb_url:str = os.getenv("MONGO_DB_URL")

create an instance of Environment class
env_variable = EnvironmentVariable()

mongo_client = pymongo.MongoClient(env_variable.mongodb_url)