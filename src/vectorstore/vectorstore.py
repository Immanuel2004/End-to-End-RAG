import sys
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from src.exception.Execption import CustomException
from src.logger.logging import logging
from dotenv import load_dotenv

load_dotenv()

class VectorStore:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings()
        self.vectorstore = None
        self.retriever = None
    
    def create_retriever(self,documents:List[Document]):
        try:
            self.vectorstore = FAISS.from_documents(documents,self.embeddings)
            self.retriever = self.vectorstore.as_retriever()
            logging.info("Created Retriever")
        except Exception as e:
            raise CustomException(e,sys)
    
    def get_retriever(self):
        try:
            if self.retriever is None:
                raise ValueError("Vectorstore not initailized . call create_vectorstore first")
        except Exception as e:
            raise CustomException(e,sys)
