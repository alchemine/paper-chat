"""LLM utility module"""

from os import environ as env

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

# from langchain_huggingface import HuggingFaceEmbeddings

from paper_chat.core.utils import SINGLETON_MANAGER
from paper_chat.core.configs import CONFIGS_LLM


############################################################
# Chat LLM
############################################################
def get_chat_llm(configs_chat_llm: dict = CONFIGS_LLM.chat_llm) -> AzureChatOpenAI:
    """Get AzureChatOpenAI instance"""
    provider = configs_chat_llm.provider
    match provider:
        case "azure":
            return get_azure_chat_llm(configs_chat_llm)

        case "huggingface":
            raise NotImplementedError("HuggingFace chat LLM is not implemented yet.")

        case _:
            raise ValueError(f"Invalid configs.llm.provider: {provider}")


def get_azure_chat_llm(configs_chat_llm: dict) -> AzureChatOpenAI:
    """Get AzureChatOpenAI instance"""
    key = "AzureChatOpenAI"
    if not hasattr(SINGLETON_MANAGER, key):
        llm = AzureChatOpenAI(
            # azure_endpoint=env["AZURE_OPENAI_ENDPOINT"],
            deployment_name=env["AZURE_OPENAI_LLM_DEPLOYMENT_NAME"],
            # openai_api_version=env["AZURE_OPENAI_API_VERSION"],
            # openai_api_key=env["AZURE_OPENAI_API_KEY"],
            model=env["AZURE_OPENAI_LLM_MODEL"],
            temperature=configs_chat_llm.temperature,
        )
        setattr(SINGLETON_MANAGER, key, llm)
    return getattr(SINGLETON_MANAGER, key)


############################################################
# Embeddings
############################################################
def get_embeddings(configs_emb: dict = CONFIGS_LLM.embeddings):
    """Get embeddings instance"""
    provider = configs_emb.provider
    match provider:
        case "azure":
            return get_azure_embeddings(configs_emb)

        case "huggingface":
            raise NotImplementedError("HuggingFace embeddings is not implemented yet.")
        #     return get_huggingface_embeddings()

        case _:
            raise ValueError(f"Invalid configs.embeddings.provider: {provider}")


def get_azure_embeddings(configs_emb: dict) -> AzureOpenAIEmbeddings:
    """Get AzureEmbeddings instance"""
    key = "AzureOpenAIEmbeddings"
    if not hasattr(SINGLETON_MANAGER, key):
        embeddings = AzureOpenAIEmbeddings(
            # azure_endpoint=env["AZURE_OPENAI_ENDPOINT"],
            deployment=env["AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT"],
            # openai_api_version=env["AZURE_OPENAI_API_VERSION"],
            # openai_api_key=env["AZURE_OPENAI_API_KEY"],
            model=env["AZURE_OPENAI_EMBEDDINGS_MODEL"],
        )
        setattr(SINGLETON_MANAGER, key, embeddings)
    return getattr(SINGLETON_MANAGER, key)


# def get_huggingface_embeddings():
#     """Get HuggingFace instance"""
#     key = "HuggingFaceEmbeddings"
#     if not hasattr(SINGLETON_MANAGER, key):
#         embeddings = HuggingFaceEmbeddings(
#             model_name=CFGS["chain"].huggingface.embeddings_model,
#             model_kwargs={"device": "cuda"},
#         )
#         setattr(SINGLETON_MANAGER, key, embeddings)
#     return getattr(SINGLETON_MANAGER, key)


############################################################
# Trace chain
############################################################
def trace_chain():
    env["LANGCHAIN_TRACING_V2"] = "true"
    env["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    # os.environ["LANGCHAIN_ENDPOINT"] = "<your-api-key>"


############################################################
# Constants
############################################################
CHAT_LLM = get_chat_llm()
EMBEDDINGS = get_embeddings()
