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

    Args:
        api_choice: The selected API provider ("openai", "groq", "huggingface").
        api_key: The user's API key
        model_name: The specific model name

    Returns:
        An initialized langChain BaseChatModel instance.

    Raises:
        ValueError: if an invalid API choice is provided
    """

    if not api_key:
        raise ValueError(f"API key for {api_choice.upper()} is required.")

    if api_choice.lower() == "openai":
        return ChatOpenAI(api_key=api_key, model=model_name)
    
    elif api_choice.lower() == "groq":
        return ChatGroq(api_key=api_key, model=model_name)
    
    elif api_choice.lower() == "huggingface":
        client = InferenceClient(token=api_key)
        return ChatHuggingFace(inference_client = client, llm = model_name)
    else:
        raise ValueError(f"Invalid API choice: {api_choice}. must be 'openai', 'groq', or 'huggingface'")
    
    