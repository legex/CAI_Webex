from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
#from dataclasses import dataclass

class DBBase:
    def __init__(self, loginurl, database: str, collection: str):
        self.uri = loginurl
        self.client = MongoClient(self.uri, server_api=ServerApi('1'))
        self.database = self.client[database]
        self.collection_name = collection
        self.collection = self.database[self.collection_name]

    @property
    def mongo_collection(self):
        return self.collection

    def _collection_exists(self):
        return self.collection_name in self.database.list_collection_names()

    def _create_data(self):
        db = self.database
        db.create_collection(self.collection_name)
        return self.mongo_collection

    def _delete_data(self):
        self.mongo_collection.drop()

# @dataclass
# class RAGConfig:
#     loginurl: str
#     database: str = ''
#     collection: str = ''
#     verbose: bool = True
