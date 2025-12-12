import os
import sys
from utils.logger import logging
from utils.exception import CustomException
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from config import CHUNK_SIZE, CHUNK_OVERLAP

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = CHUNK_SIZE,
            chunk_overlap = CHUNK_OVERLAP
        )

    def load_and_split_document(self, file_path: str, doc_id: str, filename: str) -> list[Document]:
        """ 
        Loads a document from a path, adds unique metadata, and splits it into chunks.

        Args:
            file_path: The temporary path where the uploaded file is stored
            doc_id: A unique identifier
            filename: The original name of the file
        
        Returns:
            A list of split and processed LangChain Document objects
        
        """
        logging.info(f"Loading document: {filename} with ID: {doc_id}")

        #load document using unstructured
        try:
            loader = UnstructuredFileLoader(file_path)
            documents = loader.load()
        except Exception as e:
            raise CustomException(e, sys)
        
        #Add crucial metadata
        for doc in documents:
            doc.metadata['document_id'] = doc_id
            doc.metadata['filename'] = filename
            doc.metadata['source_path'] = file_path
        
        #splits text into chunks
        chunks = self.text_splitter.split_documents(documents)

        logging.info(f"Split {filename} into {len(chunks)} chunks.")
        return chunks

        


        
        