from datamanagement.core.embedding import ChunkAndEmbed
from datamanagement.db.db_base import DBBase

class MongoDBConn(DBBase):
    def __init__(self, loginurl, source=None, weburl=None, database='', collection='', verbose=True):
        super().__init__(loginurl, database, collection)
        self.source = source
        self.urls = weburl
        self.verbose = verbose


    def _insert_chunks(self, url, query_chunks, query_embeddings, response_chunks, response_embeddings):
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
            self.mongo_collection.insert_many(docs)
            if self.verbose:
                print(f"Inserted {len(docs)} documents for: {url}")

    def save_data_to_mongo(self):
        if not self._collection_exists():
            self._create_data()
        for url in self.urls:
            try:
                if self.mongo_collection.find_one({"thread_url": url}):
                    if self.verbose:
                        print(f"URL already exists. Skipping: {url}")
                    continue
                chunk_embed = ChunkAndEmbed(self.source, url, verbose=self.verbose)
                query_embeddings, response_embeddings, query_chunks, response_chunks = chunk_embed.generate_embedding()
                self._insert_chunks(url, query_chunks, query_embeddings, response_chunks, response_embeddings)
            except Exception as e:
                print(f"Update failed for {url}: {e}")
