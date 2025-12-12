## chunks settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

## embdding model settings
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

## vector store settings
QDRANT_COLLECTION_NAME = "multi_doc_comparison_rag"
QDRANT_LOCATION = ":memory:"

## model choices
MODEL_CHOICES = {
    "openai":[
        "gpt-4o",
        "gpt-4-turbo",
        "gpt-3.5-turbo-0125"
    ],
    
    "groq": [
        "llama-3.1-8b-instant",
        
    ],
    
    "huggingface": [
        "HuggingFaceH4/zephyr-7b-beta",  # General purpose chat model
        "mistralai/Mixtral-8x7B-Instruct-v0.1" # Powerful model (requires token access)
    ],
    "default": {
        "openai": "gpt-4o",
        "groq": "llama-3.1-8b-instant",
        "huggingface": "HuggingFaceH4/zephyr-7b-beta"
    }
}

# LangSmith Configuration (optional - can also be set via UI)
LANGSMITH_API_KEY = None  # Set this or provide via UI
LANGSMITH_PROJECT = "rag-evaluation"  # Project name in LangSmith

# RAGAS Configuration
RAGAS_METRICS = [
    "faithfulness",
    "answer_relevancy", 
    "context_precision",
    "context_recall"
]