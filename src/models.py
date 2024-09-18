from typing import List
import os
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

def create_llm(with_callbacks: List[any]) -> AzureChatOpenAI:
    llm: AzureChatOpenAI = None
    if "AZURE_OPENAI_API_KEY" in os.environ:
        llm = AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_deployment=os.getenv("AZURE_OPENAI_COMPLETION_DEPLOYMENT_NAME"),
            openai_api_version=os.getenv("AZURE_OPENAI_VERSION"),
            temperature=0,
            streaming=True,
            callbacks=with_callbacks
        )
    else:
        token_provider = get_bearer_token_provider(DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")
        llm = AzureChatOpenAI(
            azure_ad_token_provider=token_provider,
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            azure_deployment=os.getenv("AZURE_OPENAI_COMPLETION_DEPLOYMENT_NAME"),
            openai_api_version=os.getenv("AZURE_OPENAI_VERSION"),
            temperature=0,
            openai_api_type="azure_ad",
            streaming=True,
            callbacks=with_callbacks
        )

    return llm

def create_embeddings_model() -> AzureOpenAIEmbeddings:
    if "AZURE_OPENAI_API_KEY" in os.environ:
        embeddings_model = AzureOpenAIEmbeddings(    
            azure_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
            openai_api_version = os.getenv("AZURE_OPENAI_VERSION"),
            model= os.getenv("AZURE_OPENAI_EMBEDDING_MODEL"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY")
        )
    else:
        token_provider = get_bearer_token_provider(DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")
        embeddings_model = AzureOpenAIEmbeddings(    
            azure_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
            openai_api_version = os.getenv("AZURE_OPENAI_VERSION"),
            model= os.getenv("AZURE_OPENAI_EMBEDDING_MODEL"),
            azure_ad_token_provider = token_provider
        )
    return embeddings_model