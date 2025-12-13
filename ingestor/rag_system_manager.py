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
from llm_api.api_manager import get_llm, get_openai_eval_llm
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
        if not doc_ids:
            logging.warning("No doc_ids provided to get_filtered_retriever, using unfiltered retriever")
            vector_store = self.get_qdrant_vectorstore()
            return vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})
        
        vector_store = self.get_qdrant_vectorstore()
        
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
    
    def _get_system_instruction(self, doc_ids: list, prompt_style: str) -> str:
        """
        Returns the appropriate system instruction based on prompt style and number of documents
        """
        
        # Validate inputs
        if not doc_ids:
            logging.warning("No document IDs provided to _get_system_instruction")
            doc_ids = ["Unknown"]
        
        if not prompt_style:
            logging.warning("No prompt_style provided, defaulting to 'Standard'")
            prompt_style = "Standard"
        
        # ============================================================
        # MULTI-DOCUMENT COMPARISON PROMPTS
        # ============================================================
        if len(doc_ids) > 1:
            if prompt_style == "Standard":
                return (
                    "You are an expert Oracle DBA with 20 years of experience analyzing AWR reports and performance data.\n\n"
                    f"You are comparing multiple documents: {', '.join(doc_ids)}\n\n"
                    "Analyze the retrieved context systematically:\n"
                    "1. Identify key performance metrics from each document\n"
                    "2. Compare and contrast the findings\n"
                    "3. Highlight differences, trends, or improvements\n"
                    "4. Point out any performance issues or anomalies\n"
                    "5. Provide actionable recommendations if applicable\n\n"
                    "IMPORTANT: Cite the Document ID (e.g., [doc_A]) for every specific data point.\n\n"
                    "Format your response with clear sections and bullet points for readability."
                )
            
            elif prompt_style == "Detailed Step-by-Step":
                return (
                    "You are an expert Oracle DBA comparing multiple AWR reports.\n\n"
                    f"Documents to analyze: {', '.join(doc_ids)}\n\n"
                    "Follow this analysis process:\n\n"
                    "**STEP 1 - Extract Key Metrics:**\n"
                    "For each document, list:\n"
                    "- CPU usage (%)\n"
                    "- DB Time\n"
                    "- Top 3 Wait Events\n"
                    "- Parse statistics\n"
                    "- Top SQL by resource consumption\n\n"
                    "**STEP 2 - Compare Metrics:**\n"
                    "Create a comparison showing:\n"
                    "- Which metrics improved?\n"
                    "- Which metrics degraded?\n"
                    "- What new issues appeared?\n\n"
                    "**STEP 3 - Trend Analysis:**\n"
                    "- Is performance improving or degrading overall?\n"
                    "- What's causing the changes?\n\n"
                    "**STEP 4 - Recommendations:**\n"
                    "Based on the comparison, suggest:\n"
                    "- Actions to address degrading metrics\n"
                    "- What's working well to continue\n\n"
                    "Always cite Document IDs (e.g., [doc_A], [doc_B]) for each data point."
                )
            
            elif prompt_style == "Issue-Focused":
                return (
                    "You are an expert Oracle DBA performing comparative analysis.\n\n"
                    f"Analyzing documents: {', '.join(doc_ids)}\n\n"
                    "Provide a structured comparison:\n\n"
                    "### 📊 EXECUTIVE SUMMARY\n"
                    "Brief overview of what changed between reports.\n\n"
                    "### 📈 METRIC COMPARISON TABLE\n"
                    "| Metric | Doc 1 | Doc 2 | Change | Status |\n"
                    "Present key metrics side-by-side.\n\n"
                    "### 📴 NEW ISSUES (appeared in later reports)\n"
                    "List any new problems that emerged.\n\n"
                    "### ✅ RESOLVED ISSUES (fixed since earlier reports)\n"
                    "List what improved.\n\n"
                    "### 🟡 ONGOING ISSUES (persist across reports)\n"
                    "List continuing problems.\n\n"
                    "### 💡 RECOMMENDATIONS\n"
                    "Based on trends, what actions should be taken?\n\n"
                    "Always cite document IDs [doc_X] for specific values."
                )
        
        # ============================================================
        # SINGLE DOCUMENT ANALYSIS PROMPTS
        # ============================================================
        else:
            if prompt_style == "Standard":
                return (
                    "You are an expert Oracle DBA with 20 years of experience analyzing AWR reports.\n\n"
                    f"Analyze the AWR report from document [{doc_ids[0]}] systematically:\n\n"
                    "**Step 1: Key Metrics Analysis**\n"
                    "- Identify critical metrics (CPU, DB Time, Wait Events, etc.)\n"
                    "- Compare against Oracle best practices\n\n"
                    "**Step 2: Issue Identification**\n"
                    "- List any metrics outside normal ranges\n"
                    "- Categorize by severity (Critical/Warning/Info)\n\n"
                    "**Step 3: Root Cause Analysis**\n"
                    "- For each issue, explain the likely root cause\n"
                    "- Reference specific evidence from the report\n\n"
                    "**Step 4: Solutions**\n"
                    "- Provide specific, actionable recommendations\n"
                    "- Prioritize by impact (High/Medium/Low)\n"
                    "- Include implementation effort (Easy/Medium/Hard)\n\n"
                    "Be specific with metric values and cite the Document ID for your sources."
                )
            
            elif prompt_style == "Detailed Step-by-Step":
                return (
                    "You are an expert Oracle DBA analyzing an AWR report.\n\n"
                    f"Document: [{doc_ids[0]}]\n\n"
                    "Analyze this AWR report step-by-step:\n\n"
                    "**STEP 1 - Metric Extraction:**\n"
                    "List the key metrics you find:\n"
                    "- CPU usage and trend\n"
                    "- DB Time breakdown\n"
                    "- Top Wait Events (name and % of DB time)\n"
                    "- Parse statistics (hard vs soft)\n"
                    "- Buffer cache hit ratio\n"
                    "- Top 3 SQL statements by elapsed time\n\n"
                    "**STEP 2 - Threshold Comparison:**\n"
                    "For each metric, compare against best practices:\n"
                    "- CPU: Should be < 80%\n"
                    "- Parse ratio: Hard parses < 10% of total\n"
                    "- Buffer cache: Should be > 95%\n"
                    "- Wait events: No single event should dominate > 50%\n\n"
                    "**STEP 3 - Issue Identification:**\n"
                    "List issues found:\n"
                    "📴 CRITICAL: [issues requiring immediate attention]\n"
                    "🟡 WARNING: [issues needing investigation]\n"
                    "ℹ️ INFO: [observations and recommendations]\n\n"
                    "**STEP 4 - Root Cause & Solutions:**\n"
                    "For each issue:\n"
                    "- Root Cause: [Why is this happening?]\n"
                    "- Impact: [What's affected?]\n"
                    "- Solution: [Specific steps to resolve]\n"
                    "- Priority: [High/Medium/Low]\n"
                    "- Effort: [Easy/Medium/Hard]\n\n"
                    "Show your reasoning for each step. Be specific with values and cite document ID."
                )
            
            elif prompt_style == "Issue-Focused":
                return (
                    "You are an expert Oracle DBA analyzing an AWR performance report.\n\n"
                    f"Document: [{doc_ids[0]}]\n\n"
                    "Provide a comprehensive analysis in this format:\n\n"
                    "### 📊 EXECUTIVE SUMMARY\n"
                    "One paragraph: Overall health, main findings, severity level.\n\n"
                    "### 📴 CRITICAL ISSUES (Immediate Attention Required)\n"
                    "For each critical issue:\n"
                    "**Issue:** [Name]\n"
                    "- **Metric:** [Specific value vs. expected]\n"
                    "- **Root Cause:** [Why this is happening]\n"
                    "- **Impact:** [What's affected]\n"
                    "- **Solution:** [Specific fix with steps]\n"
                    "- **Priority:** High | **Effort:** Easy/Medium/Hard\n\n"
                    "### 🟡 WARNINGS (Investigation Recommended)\n"
                    "[Same format as above]\n\n"
                    "### ℹ️ OBSERVATIONS (Optimization Opportunities)\n"
                    "[Same format as above]\n\n"
                    "### 🎯 TOP 3 ACTION ITEMS\n"
                    "1. [Most important action with expected result]\n"
                    "2. [Second priority]\n"
                    "3. [Third priority]\n\n"
                    "### 📈 KEY METRICS SUMMARY\n"
                    "- CPU Usage: [value and assessment]\n"
                    "- DB Time: [value and assessment]\n"
                    "- Top Wait Event: [name and % of time]\n"
                    "- Buffer Cache Hit: [ratio and assessment]\n"
                    "- Parse Efficiency: [hard/soft ratio]\n\n"
                    "Be specific with numbers. Cite document ID for sources."
                )
        
        # Fallback (should rarely reach here)
        logging.warning(f"Unrecognized prompt_style: {prompt_style}. Using default.")
        return "You are an expert DBA assistant. Answer based on the provided context."

    
    def answer_query(self, question: str, doc_ids: list, api_choice: str, api_key: str, model_name: str,
                     enable_evaluation: bool = False, ground_truth: str = None, openai_eval_key: str = None,
                     prompt_style: str = "Standard") -> dict:
        """  
        Executes the RAG chain for both single-document queries and multi-document comparisons.
        
        Args:
            question: The user's question
            doc_ids: List of document IDs to search within
            api_choice: LLM API provider
            api_key: API key for the LLM
            model_name: Model name to use
            enable_evaluation: Whether to run RAGAS evaluation
            ground_truth: Ground truth answer for evaluation
            openai_eval_key: OpenAI API key for RAGAS evaluation
            prompt_style: Style of system prompt (Standard, Detailed Step-by-Step, Issue-Focused)
        
        Returns:
            dict: Contains answer, contexts, and optionally evaluation scores
        """
        logging.info("="*50)
        logging.info("ANSWERING QUERY")
        logging.info(f"Question: {question}")
        logging.info(f"Document IDs: {doc_ids}")
        logging.info(f"LLM: {api_choice} - {model_name}")
        logging.info(f"Evaluation enabled: {enable_evaluation}")
        
        # Validate inputs
        if not question or not isinstance(question, str) or not question.strip():
            logging.error("Invalid question provided")
            return {"answer": "Error: Question cannot be empty", "error": True}
        
        if not doc_ids or not isinstance(doc_ids, list):
            logging.error("Invalid doc_ids provided")
            return {"answer": "Error: doc_ids must be a non-empty list", "error": True}
        
        try:
            llm = get_llm(api_choice, api_key, model_name)
            logging.info("LLM initialized")
        except Exception as e:
            logging.error(f"LLM initialization failed: {e}")
            return {"answer": f"LLM initialization Error: {e}", "error": True}
        
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
                        "answer": "No relevant documents found in the collection",
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
        
        system_instruction = self._get_system_instruction(doc_ids, prompt_style)
        
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
                    llm=ragas_llm
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
            raise CustomException(e, sys) 

    def evaluate_with_ragas(self, question: str, answer: str, contexts: list,
                            ground_truth: str = None, llm=None) -> dict:
        """  
        Evaluate RAG response using RAGAS metrics.
        
        Args:
            question: The user's question
            answer: The generated answer
            contexts: List of retrieved context chunks
            ground_truth: Optional ground truth answer
            llm: LangChain LLM instance for evaluation
        
        Returns:
            dict: RAGAS evaluation scores or error information
        """
        from ragas.llms import LangchainLLMWrapper
        
        try:
            if not question or not answer or not contexts:
                logging.warning("Missing required evaluation data")
                return {"error": "Missing question, answer, or contexts for evaluation"}
            
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

            # RAGAS expects its own LLM wrapper
            if not llm:
                logging.error("No LLM provided for RAGAS evaluation")
                return {"error": "LLM instance required for RAGAS evaluation"}

            result = evaluate(
                dataset,
                metrics=metrics,
                llm=LangchainLLMWrapper(llm),
                embeddings=self.embeddings,
            )
            
            # Convert to dict and round scores
            scores = result.to_pandas().iloc[0].to_dict()
            scores_out = {k: round(v, 4) for k, v in scores.items() if isinstance(v, (int, float))}
            
            logging.info(f"RAGAS Evaluation Scores: {scores_out}")
            return scores_out
        
        except Exception as e:
            logging.error(f"RAGAS evaluation failed: {e}", exc_info=True)
            return {"error": str(e)}
        
    def get_evaluation_summary(self) -> dict:
        """
        Get summary of all evaluation results.
        
        Returns:
            dict: Summary statistics and all evaluation results
        """
        if not self.evaluation_results:
            return {"message": "No evaluations run yet"}
        
        # Calculate average scores
        all_scores = {}
        for result in self.evaluation_results:
            # Validate result structure
            if not isinstance(result, dict):
                logging.warning(f"Invalid result format: {type(result)}")
                continue
            
            scores = result.get("scores", {})
            if not isinstance(scores, dict):
                logging.warning(f"Invalid scores format in result: {type(scores)}")
                continue
            
            for metric, score in scores.items():
                if metric != "error" and isinstance(score, (int, float)):
                    if metric not in all_scores:
                        all_scores[metric] = []
                    all_scores[metric].append(score)
        
        avg_scores = {
            metric: round(sum(scores) / len(scores), 4)
            for metric, scores in all_scores.items()
            if len(scores) > 0
        }

        return {
            "total_evaluations": len(self.evaluation_results),
            "average_scores": avg_scores,
            "all_results": self.evaluation_results
        }