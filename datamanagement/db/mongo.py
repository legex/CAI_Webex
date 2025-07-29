import pymongo.errors
from datamanagement.core.embedding import ChunkAndEmbed
from datamanagement.db.db_base import DBBase
from datamanagement.core.logger import setup_logger

logger = setup_logger('mongodb_conn', 'datamanagement/log/mongodb_conn.log')

class MongoDBConn(DBBase):
    """
    MongoDB connection handler for inserting chunked and embedded documents into MongoDB.

    Extends DBBase for MongoDB connection and collection handling.
    Uses ChunkAndEmbed to generate text chunks and embeddings.
    """
    def __init__(self, loginurl,
                 source=None, weburl=None,
                 database='', collection='',
                 verbose=True
                 ):
        """
        Initialize MongoDBConn with source info, list of URLs, and verbosity.

        Args:
            loginurl (str): MongoDB connection URI.
            source (str, optional): Source identifier (e.g., "webex", "community").
            weburl (list[str], optional): List of URLs to process.
            database (str): MongoDB database name.
            collection (str): MongoDB collection name.
            verbose (bool, optional): If True, enables print/log output.
        """
        super().__init__(loginurl, database, collection)
        self.source = source
        self.urls = weburl
        self.verbose = verbose
        logger.info("MongoDBConn initialized for source=%s with %d URLs.",
                    source, len(weburl) if weburl else 0)


    def _insert_chunks(self,
                       url,
                       query_chunks,
                       query_embeddings,
                       response_chunks,
                       response_embeddings
                       ):
        """
        Insert chunk-embedding document pairs into MongoDB collection.

        Args:
            url (str): Thread URL for the chunks.
            query_chunks (List[str]): List of query text chunks.
            query_embeddings (List[List[float]]): Corresponding embeddings for query chunks.
            response_chunks (List[str]): List of response text chunks.
            response_embeddings (List[List[float]]): Corresponding embeddings for response chunks.
        """
        docs = []
        for _, (q_chunk, q_emb) in enumerate(zip(query_chunks, query_embeddings)):
            for _, (r_chunk, r_emb) in enumerate(zip(response_chunks, response_embeddings)):
                docs.append({
                    "thread_url": url,
                    "source": self.source,
                    "query_chunk": q_chunk,
                    "query_embedding": q_emb,
                    "response_chunk": r_chunk,
                    "response_embedding": r_emb,
                })
        if docs:
            try:
                self.mongo_collection.insert_many(docs)
                logger.info("Inserted %d documents for URL: %s", len(docs), url)
                if self.verbose:
                    print(f"Inserted {len(docs)} documents for: {url}")
            except pymongo.errors.PyMongoError as e:
                logger.error("Failed to insert documents for %s: %s", url, e)
                if self.verbose:
                    print(f"Failed to insert documents for {url}: {e}")

    def save_data_to_mongo(self):
        """
        Main method to save data to MongoDB.

        - Checks if collection exists; creates if missing.
        - Iterates over URLs and processes each if not already present.
        - For each URL, scrapes and generates embeddings and chunks,
          then inserts them into MongoDB.
        """
        if not self._collection_exists():
            self._create_data()
            logger.info("Collection '%s' created.", self.collection_name)

        if not self.urls:
            logger.warning("No URLs provided to save_data_to_mongo.")
            if self.verbose:
                print("No URLs provided to save_data_to_mongo.")
            return

        for url in self.urls:
            try:
                if self.mongo_collection.find_one({"thread_url": url}):
                    logger.info("URL already exists in collection, skipping: %s", url)
                    if self.verbose:
                        print(f"URL already exists. Skipping: {url}")
                    continue
                chunk_embed = ChunkAndEmbed(self.source, url, verbose=self.verbose)
                query_embeddings, response_embeddings, query_chunks, response_chunks = chunk_embed.generate_embedding()
                self._insert_chunks(url, query_chunks,
                                    query_embeddings,
                                    response_chunks,
                                    response_embeddings)
            except pymongo.errors.PyMongoError as e:
                logger.error("MongoDB error for %s: %s", url, e)
                if self.verbose:
                    print(f"MongoDB error for {url}: {e}")
            except (ValueError, TypeError) as e:
                logger.error("Data processing error for %s: %s", url, e)
                if self.verbose:
                    print(f"Data processing error for {url}: {e}")
