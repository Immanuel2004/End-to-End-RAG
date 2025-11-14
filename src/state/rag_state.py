import sys
from src.exception.Execption import CustomException
from src.logger.logging import logging

from typing import List
from pydantic import BaseModel
from langchain_core.documents import Document

class RAGState(BaseModel):
    question : str
    retrieved_docs :List[Document] = []
    answer : str = ""