import os
import sys
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from qdrant_client import QdrantClient, models
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from config import EMBEDDING_MODEL_NAME, QDRANT_COLLECTION_NAME, QDRANT_LOCATION
from llm_api.api_manager import get_llm, get_openai_eval_llm # <-- ADDED get_openai_eval_llm
from utils.logger import logging
from utils.exception import CustomException
from sentence_transformers import SentenceTransformer

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from datasets import Dataset

# Import for Qdrant filtering
from qdrant_client.models import Filter, FieldCondition, MatchAny

class RAGSystemManager:
    """ 
    The core object managing embeddings, Qdrant vector store, ingestion, 
    retrieval filtering and the final RAG chain execution.
    """

    def __init__(self, qdrant_location=QDRANT_LOCATION, collection_name=QDRANT_COLLECTION_NAME):
        logging.info("="*50)
        logging.info("Initializing RAG System Manager (LangSmith disabled)...")
        logging.info(f"Qdrant location: {qdrant_location}")
        logging.info(f"Collection name: {collection_name}")
        logging.info(f"Embedding model: {EMBEDDING_MODEL_NAME}")

        # Initialize evaluation results storage
        self.evaluation_results = []  


        try: 
            logging.info("Loading SentenceTransformer model...")
            self._model_client = SentenceTransformer(EMBEDDING_MODEL_NAME)
            self.vector_size = self._model_client.get_sentence_embedding_dimension()
            logging.info(f"Model loaded. Vector size: {self.vector_size}")
        except Exception as e:
            logging.error(f"FATAL: SentenceTransformer initialization failed. Error: {e}")
            raise CustomException(e, sys)
        
        # Initialize embedding model
        logging.info("Initializing HuggingFace embeddings...")
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        logging.info("Embeddings initialized")

        # Initialize Qdrant Client
        logging.info(f"Connecting to Qdrant at: {qdrant_location}")
        self.qdrant_client = QdrantClient(location=qdrant_location)
        self.collection_name = collection_name
        self.qdrant_location = qdrant_location
        logging.info("Qdrant client connected")
        
        self._ensure_collection_exists()
        logging.info("="*50)

    def _ensure_collection_exists(self):
        """ Creates the Qdrant collection if it does not exist """
        
        vector_size = self.vector_size
        logging.info(f"Checking if collection '{self.collection_name}' exists...")
               
        try:
            # Check if collection exists
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            logging.info(f"Collection '{self.collection_name}' exists with {collection_info.points_count} points")
        except Exception as e:
            # Create collection if it doesn't exist
            logging.info(f"Collection not found. Creating new collection...")
            logging.info(f"Vector config: size={vector_size}, distance=COSINE")
            
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE
                )
            )
            logging.info(f"Created collection '{self.collection_name}' with vector size {vector_size}")

    def get_qdrant_vectorstore(self) -> QdrantVectorStore:
        """ Returns the LangChain Qdrant vectorstore object for retrieval """
        logging.info("Creating QdrantVectorStore with existing client...")
        
        vectorstore = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=self.collection_name,
            embedding=self.embeddings,
        )
        
        logging.info("VectorStore created")
        return vectorstore        
    
    def index_document_chunks(self, processed_chunks: list[Document]):
        """  
        Generates embeddings and adds document chunks to the Qdrant collection
        """
        logging.info("="*50)
        logging.info("INDEXING DOCUMENTS")
        
        try:
            if not processed_chunks:
                logging.warning("No chunks provided for indexing")
                return 
            
            logging.info(f"Received {len(processed_chunks)} chunks to index")
            logging.info(f"Document: {processed_chunks[0].metadata.get('filename', 'Unknown')}")
            logging.info(f"Document ID: {processed_chunks[0].metadata.get('document_id', 'Unknown')}")
            logging.info(f"Sample chunk metadata: {processed_chunks[0].metadata}")
            logging.info(f"Sample content (first 200 chars): {processed_chunks[0].page_content[:200]}")
            
            # Check collection before indexing
            collection_before = self.qdrant_client.get_collection(self.collection_name)
            points_before = collection_before.points_count
            logging.info(f"Collection has {points_before} points before indexing")

            # Use add_documents with the existing vectorstore
            logging.info("Getting existing vectorstore and adding documents...")
            vector_store = self.get_qdrant_vectorstore()
            ids = vector_store.add_documents(documents=processed_chunks)
            logging.info(f"add_documents returned IDs: {ids[:3] if ids else 'None'}... (showing first 3)")

            # Verify indexing
            collection_after = self.qdrant_client.get_collection(self.collection_name)
            points_after = collection_after.points_count
            points_added = points_after - points_before
            
            logging.info(f"Collection now has {points_after} points (added {points_added} points)")
            
            if points_added == 0:
                logging.error("NO POINTS WERE ADDED! Documents were not indexed!")
                raise Exception("Documents were not indexed - no points added to collection!")
            
            if points_added != len(processed_chunks):
                logging.warning(f"Warning: Expected to add {len(processed_chunks)} points but added {points_added}")

            logging.info(f"Successfully indexed {points_added} chunks for {processed_chunks[0].metadata['filename']}")
            logging.info("="*50)
            
        except Exception as e:
            logging.error(f"Qdrant Indexing Failed: {e}", exc_info=True)
            logging.info("="*50)
            raise CustomException(e, sys) 
        
    def get_collection_stats(self):
        """Get statistics about the current collection"""
        try:
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            points_count = collection_info.points_count
            
            # Get sample points
            sample_points = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                limit=5,
                with_payload=True,
                with_vectors=False
            )[0]
            
            return {
                "points_count": points_count,
                "sample_metadata": [p.payload for p in sample_points] if sample_points else []
            }
        except Exception as e:
            logging.error(f"Error getting collection stats: {e}")
            return {"error": str(e)}
    
    def get_filtered_retriever(self, doc_ids: list, k: int = 5):
        """  
        Creates a LangChain retriever filtered to only search within the specified document IDs
        """
        vector_store = self.get_qdrant_vectorstore()

        # from qdrant_client.models import Filter, FieldCondition, MatchAny # Already imported
        
        logging.info(f"Creating retriever for doc_ids: {doc_ids}")
        
        qdrant_filter = Filter(
            must=[
                FieldCondition(
                    key="metadata.document_id",
                    match=MatchAny(any=doc_ids)
                )
            ]
        )
        
        return vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k, "filter": qdrant_filter}
        )
    
    def answer_query(self, question: str, doc_ids: list, api_choice: str, api_key: str, model_name: str,
                     enable_evaluation: bool = False, ground_truth: str = None, openai_eval_key: str = None) -> dict:  # <-- ALWAYS RETURN DICT
        """  
        Executes the RAG chain for both single-document queries and multi-document comparisons
        """
        logging.info("="*50)
        logging.info("ANSWERING QUERY")
        logging.info(f"Question: {question}")
        logging.info(f"Document IDs: {doc_ids}")
        logging.info(f"LLM: {api_choice} - {model_name}")
        logging.info(f"Evaluation enabled: {enable_evaluation}")
        
        try:
            llm = get_llm(api_choice, api_key, model_name)
            logging.info("LLM initialized")
        except Exception as e:
            logging.error(f"LLM initialization failed: {e}")
            return {"answer": f"LLM initialization Error: {e}", "error": True}  # <-- RETURN DICT
        
        # Test retrieval first
        try:
            logging.info(f"Creating filtered retriever for doc_ids: {doc_ids}")
            retriever = self.get_filtered_retriever(doc_ids=doc_ids, k=10)
            
            logging.info("Testing document retrieval...")
            test_docs = retriever.invoke(question)
            logging.info(f"Retrieved {len(test_docs)} documents")
            
            if test_docs:
                logging.info("Sample retrieved document:")
                logging.info(f"  - Doc ID: {test_docs[0].metadata.get('document_id', 'Unknown')}")
                logging.info(f"  - Filename: {test_docs[0].metadata.get('filename', 'Unknown')}")
                logging.info(f"  - Content preview: {test_docs[0].page_content[:200]}")
            
            if not test_docs:
                logging.warning("No documents retrieved with filter! Trying without filter...")
                vector_store = self.get_qdrant_vectorstore()
                retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 10})
                test_docs = retriever.invoke(question)
                logging.info(f"Without filter, retrieved {len(test_docs)} documents")
                
            if not test_docs:
                logging.error("No relevant documents found even without filter!")
                return {
                    "answer": " No relevant documents found",
                    "contexts": [],
                    "error": True
                }
                
        except Exception as e:
            logging.error(f"Retrieval test failed: {e}", exc_info=True)
            return {"answer": f"Retrieval Error: {e}", "error": True}
        
        # Store contexts for evaluation
        contexts = [doc.page_content for doc in test_docs]

        # Define prompt based on comparison requirement
        logging.info(f"Building RAG chain for {len(doc_ids)} document(s)...")
        
        if len(doc_ids) > 1:
            system_instruction = (
                "You are an expert DBA, comparison and synthesis bot. The user has asked a question "
                "about multiple reports. Analyze the retrieved context from "
                f"documents {', '.join(doc_ids)} and provide a structured answer that **compares and contrasts** "
                "the key points based on the user's question. You MUST cite the Document ID (e.g., [doc_A]) "
                "for every specific data point you mention to show which report it came from."
            )
        else:
            system_instruction = (
                "You are an expert DBA assistant. Answer the user's question only "
                "based on the provided context. If the answer is not in the context, state that. "
                f"The context is from document {doc_ids[0]}. Cite the Document ID ([{doc_ids[0]}]) for your source."
            )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_instruction + "\n\nContext: {context}"),
            ("user", "{question}")
        ])

        # Define the RAG chain
        def format_docs(docs):
            """ Formats retrieved documents for the LLM prompt """
            return "\n---\n".join([
                f"Document ID: {doc.metadata.get('document_id', 'Unknown')}\nFilename: {doc.metadata.get('filename', 'Unknown')}\nContent: {doc.page_content}"
                for doc in docs
            ])

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm 
            | StrOutputParser()
        )

        # Invoke and return the result
        try:
            logging.info("Invoking RAG chain...")
            answer = rag_chain.invoke(question)  
            logging.info("RAG chain executed successfully")
            logging.info(f"Response length: {len(answer)} characters")  

            result = {
                "answer": answer,
                "contexts": contexts,
                "retrieved_docs_count": len(test_docs),
                "error": False
            }

            # Run RAGAS evaluation if enabled
            if enable_evaluation:
                logging.info("Running RAGAS evaluation...")

                # RAGAS REQUIRES AN OPENAI LLM (OR COMPATIBLE) FOR METRICS
                if not openai_eval_key:
                    logging.error("No OpenAI API key provided for RAGAS evaluation")
                    result["evaluation"] = {"error": "OpenAI API key required for RAGAS evaluation"}
                    logging.info("="*50)
                    return result

                # Initialize dedicated LLM for RAGAS
                try:
                    ragas_llm = get_openai_eval_llm(openai_eval_key)
                except Exception as e:
                    logging.error(f"RAGAS LLM initialization failed: {e}")
                    result["evaluation"] = {"error": f"RAGAS LLM initialization failed: {e}"}
                    logging.info("="*50)
                    return result


                logging.info(f"Using OpenAI API for RAGAS evaluation")

                eval_scores = self.evaluate_with_ragas(
                    question=question,
                    answer=answer,
                    contexts=contexts,
                    ground_truth=ground_truth,
                    llm=ragas_llm # Pass the dedicated RAGAS LLM
                )
                result["evaluation"] = eval_scores
                self.evaluation_results.append({
                    "question": question,
                    "answer": answer,
                    "scores": eval_scores
                })
                logging.info(f"Evaluation complete: {eval_scores}")
            
            logging.info("="*50)
            return result
            
        except Exception as e:
            logging.error(f"RAG Chain Invocation Failed: {e}", exc_info=True)
            logging.info("="*50)
            # Re-raise as CustomException for consistent error handling
            raise CustomException(e, sys) 

    def evaluate_with_ragas(self, question: str, answer: str, contexts: list,
                            ground_truth: str = None, llm=None) -> dict:
        """  
        Evaluate RAG response using RAGAS metrics
        """
        from ragas.llms import LangchainLLMWrapper # Local import for this function
        
        try:
            # Prepare data for RAGAS
            data = {
                "question": [question],
                "answer": [answer],
                "contexts": [contexts]
            }

            # Add ground truth if provided
            if ground_truth:
                data["ground_truth"] = [ground_truth]

            dataset = Dataset.from_dict(data)
            dataset = dataset.flatten()

            # Select metrics based on available data
            metrics = [
                faithfulness,
                answer_relevancy
            ]

            if ground_truth:
                metrics.extend([context_precision, context_recall])

            # RAGAS expects its own LLM wrapper for non-OpenAI LLMs, 
            # or a direct OpenAI client. Here we pass a dedicated LangChain OpenAI LLM.
            result = evaluate(
                dataset,
                metrics=metrics,
                llm=LangchainLLMWrapper(llm), # Wrap the dedicated OpenAI LLM for RAGAS
                embeddings=self.embeddings,
                
            )
            
            # Convert to dict and round scores
            scores = result.to_pandas().iloc[0].to_dict()

            scores_out = {k: round(v, 4) for k, v in scores.items() if isinstance(v, (int, float))}
            logging.info(f"RAGAS Evaluation Scores for query: {scores_out}")
            return scores_out
        
        except Exception as e:
            logging.error(f"RAGAS evaluation failed: {e}", exc_info=True)
            return {"error": str(e)}
        
    def get_evaluation_summary(self) -> dict:
        """Get summary of all evaluation results"""
        if not self.evaluation_results:
            return {"message": "No evaluations run yet"}
        
        # Calculate average scores
        all_scores = {}
        for result in self.evaluation_results:
            # Check for "scores" key and ensure it's a dict
            if isinstance(result, dict) and "scores" in result and isinstance(result["scores"], dict):
                for metric, score in result["scores"].items():
                    if metric != "error" and isinstance(score, (int, float)):
                        if metric not in all_scores:
                            all_scores[metric] = []
                        all_scores[metric].append(score)
        
        avg_scores = {
            metric: round(sum(scores) / len(scores), 4)
            for metric, scores in all_scores.items()
        }

        return {
            "total_evaluations": len(self.evaluation_results),
            "average_scores": avg_scores,
            "all_results": self.evaluation_results
        }