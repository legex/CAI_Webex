from core.generatebase import ChunkandGenerate
from core.embedding_model import EmbeddingModel
from core.logger import setup_logger

logger = setup_logger('pdfembeddings', 'log/pdfembeddings.log')

class PDFEmbed(ChunkandGenerate):
    """
    Wrapper to chunk text and generate sentence-transformer embeddings.
    """
    def __init__(self, source, url, verbose: bool = False):
        super().__init__()
        self.source = source
        self.url = url
        self.verbose = verbose
        model_wrapper = EmbeddingModel.get_instance()
        self.model = model_wrapper.model
        self.model_lock = model_wrapper.lock
        logger.info("ChunkAndEmbed initialized with source=%s, url=%s", source, url)

    def generate_embedding(self, query = None, response_text = None):

        if not query or not response_text:
            logger.error("Scraping returned empty content.")
            raise ValueError("Scraping returned empty content.")

        query_chunks = self.chunk_text(query)
        response_chunks = self.chunk_text(response_text)

        if self.verbose:
            logger.info("Generated %d query chunks, %d response chunks.",
                        len(query_chunks),
                        len(response_chunks)
                        )

        with self.model_lock:
            query_embeddings = [ self.model.encode(chunk).tolist() for chunk in query_chunks ]
            response_embeddings = [ self.model.encode(chunk).tolist() for chunk in response_chunks ]

        logger.debug("Embeddings generated successfully.")
        return query_embeddings, response_embeddings, query_chunks, response_chunks
