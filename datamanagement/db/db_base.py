from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from datamanagement.core.logger import setup_logger

logger = setup_logger('db_base', 'datamanagement/log/db_base.log')

class DBBase:
    """
    Base class to manage MongoDB connections and collection operations.

    Provides methods to connect to a MongoDB collection, check existence,
    create collections, and delete data (drop collection).
    """
    def __init__(self, loginurl, database: str, collection: str):
        """
        Initialize MongoDB client and set database and collection references.

        Args:
            loginurl (str): MongoDB connection URI.
            database (str): Database name.
            collection (str): Collection name within the database.
        """
        self.uri = loginurl
        try:
            self.client = MongoClient(self.uri, server_api=ServerApi('1'))
            self.database = self.client[database]
            self.collection_name = collection
            self.collection = self.database[self.collection_name]
            logger.info("Connected to MongoDB at %s, database '%s', collection '%s'",
                       self.uri, database, collection)
        except Exception as e:
            logger.error("Failed to connect to MongoDB: %s", e)
            raise

    @property
    def mongo_collection(self):
        """
        Return the MongoDB collection object.

        Returns:
            pymongo.collection.Collection: MongoDB collection.
        """
        return self.collection

    def _collection_exists(self):
        """
        Check if the collection exists in the database.

        Returns:
            bool: True if collection exists, False otherwise.
        """
        return self.collection_name in self.database.list_collection_names()

    def _create_data(self):
        """
        Create the collection in the database.

        Returns:
            pymongo.collection.Collection: The newly created collection.
        """
        db = self.database
        db.create_collection(self.collection_name)
        return self.mongo_collection

    def _delete_data(self):
        """
        Drop (delete) the collection from the database.
        """
        self.mongo_collection.drop()
