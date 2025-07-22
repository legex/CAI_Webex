from datamanagement.core.factory import ScraperFactory
from datamanagement.core.generatebase import ChunkandGenerate
from datamanagement.core.embedding_model import EmbeddingModel
import csv
import os

class ChunkAndEmbed(ChunkandGenerate):
    """
    Wrapper to chunk text and generate sentence-transformer embeddings.
    """
    def __init__(self, source, url, verbose: bool = False):
        super().__init__()
        self.source = source
        self.url = url
        self.verbose = verbose
        self.scraper = ScraperFactory.get_scraper(source, url)
        model_wrapper = EmbeddingModel.get_instance()
        self.model = model_wrapper.model
        self.model_lock = model_wrapper.lock

    def generate_embedding(self, query = None):
        query_text, response_text = self.scraper.scrape()

        if not query_text or not response_text:
            raise ValueError("Scraping returned empty content.")

        self.save_raw_text_pair(query_text, response_text)

        query_chunks = self.chunk_text(query_text)
        response_chunks = self.chunk_text(response_text)

        if self.verbose:
            print(f"Generated {len(query_chunks)} query chunks, {len(response_chunks)} response chunks.")

        with self.model_lock:
            query_embeddings = [ self.model.encode(chunk).tolist() for chunk in query_chunks ]
            response_embeddings = [ self.model.encode(chunk).tolist() for chunk in response_chunks ]

        return query_embeddings, response_embeddings, query_chunks, response_chunks


    def save_raw_text_pair(self, query_text, response_text, csv_filepath='scraped_pairs.csv'):
        file_exists = os.path.isfile(csv_filepath)
        with open(csv_filepath, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(
                csvfile,
                fieldnames=['query_text', 'response_text'],
                quoting=csv.QUOTE_ALL  # Quote all fields to prevent issues
            )
            if not file_exists:
                writer.writeheader()
            writer.writerow({
                'query_text': query_text,
                'response_text': response_text
            })
