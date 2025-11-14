import sys
from typing import List,Union
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pathlib import Path
from langchain_community.document_loaders import (
    WebBaseLoader,
    PyPDFLoader,
    TextLoader,
    PyPDFDirectoryLoader
)
from src.exception.Execption import CustomException
from src.logger.logging import logging

class DocumentProcessor:
    def __init__(self,chunk_size:int = 500, chunk_overlap:int=50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def load_from_urls(self,url:str)->List[Document]:
        try:
            loader = WebBaseLoader(url)
            logging.info("Loaded urls")
            return loader.load()
        except Exception as e:
            raise CustomException(e,sys)
    
    def load_from_pdf_dir(self,directory:Union[str,Path])->List[Document]:
        try:
            loader = PyPDFDirectoryLoader(str(directory))
            logging.info("Loaded PDF directory")
            return loader.load()
        except Exception as e:
            raise CustomException(e,sys)
        
    def load_from_text(self,file_path:Union[str,Path]) -> List[Document]:
        try:
            loader = TextLoader(str(file_path),encoding='utf-8')
            logging.info("Loaded Text")
            return loader.load()
        except Exception as e:
            raise CustomException(e,sys)
        
    def load_from_pdf(self,file_path:Union[str,Path])-> List[Document]:
        try:
            loader = PyPDFDirectoryLoader(str("data"))
            logging.info("Loading the data directory")
            return loader.load()
        except Exception as e:
            raise CustomException(e,sys)

    def load_documents(self,source:List[str])->List[Document]:
        try:
            docs:List[Document] = []
            for src in source:
                if src.startswith('https://') or src.startswith('http://'):
                    docs.extend(self.load_from_urls(src))
                
                path = Path('data')
                if path.is_dir:
                    docs.extend(self.load_from_pdf_dir(path))
                elif path.suffix.lower() == '.txt':
                    docs.extend(self.load_from_text(path))
                else:
                    raise ValueError(
                        f"Unsupported source type: {src}."
                        "Use URL , .txt file, or directory."
                    )
            return docs
        except Exception as e:
            raise CustomException(e,sys)
        
    def split_documents(self,documents:List[Document])->List[Document]:
        try:
            return self.splitter.split_documents(documents)
        except Exception as e:
            raise CustomException(e,sys)
        
    def process_url(self,urls:List[str])->List[Document]:
        try:
            docs = self.load_documents(urls)
            return self.split_documents(docs)
        except Exception as e:
            raise CustomException(e,sys)
