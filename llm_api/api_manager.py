from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_community.chat_models import ChatHuggingFace
from huggingface_hub import InferenceClient
from langchain.chat_models.base import BaseChatModel
import os
import sys
from utils.logger import logging
from utils.exception import CustomException

def get_llm(api_choice: str, api_key: str, model_name: str) -> BaseChatModel:
    """  
    Returns the appropriate LangChain chat model instance based on the user's choice.
    """

    if not api_key:
        raise ValueError(f"API key for {api_choice.upper()} is required.")

    if api_choice.lower() == "openai":
        return ChatOpenAI(api_key=api_key, model=model_name)
    
    elif api_choice.lower() == "groq":
        return ChatGroq(api_key=api_key, model=model_name)
    
    elif api_choice.lower() == "huggingface":
        # Pass a specific model name for the ChatHuggingFace wrapper. 
        # The original code passed 'llm = model_name', which is correct for this wrapper.
        client = InferenceClient(token=api_key)
        return ChatHuggingFace(inference_client = client, llm = model_name)
    else:
        raise ValueError(f"Invalid API choice: {api_choice}. must be 'openai', 'groq', or 'huggingface'")

def get_openai_eval_llm(api_key: str, model_name: str = "gpt-4-turbo-preview") -> BaseChatModel:
    """
    Returns a dedicated ChatOpenAI instance for RAGAS evaluation.
    RAGAS metrics are highly dependent on powerful LLMs like gpt-4.
    """
    if not api_key:
        raise ValueError("OpenAI API key for RAGAS evaluation is required.")
    
    # Use a powerful model for the evaluation LLM for best results
    return ChatOpenAI(api_key=api_key, model=model_name)