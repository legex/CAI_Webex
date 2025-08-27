from typing import Optional
from pydantic import BaseModel

class CleanRaw(BaseModel):
    """
    Pydantic model for parsing the input query for the /invoke endpoint.
    """
    rawstrings: str

class Query(BaseModel):
    query: str


class VectorSearch(BaseModel):
    """
    Pydantic model for parsing the input query for the /invoke endpoint.
    """
    querystr: str