"""LLM utility module"""

from os import environ as env

from langchain_openai import (
    ChatOpenAI,
    OpenAIEmbeddings,
    AzureChatOpenAI,
    AzureOpenAIEmbeddings,
)

# from langchain_huggingface import HuggingFaceEmbeddings

from paper_chat.core.utils import SINGLETON_MANAGER
from paper_chat.core.configs import CONFIGS_LLM


LLM_KEY = "llm"
EMBEDDINGS_KEY = "embeddings"

############################################################
# Chat LLM
############################################################
# def get_chat_llm(configs_chat_llm: dict = CONFIGS_LLM.chat_llm) -> AzureChatOpenAI:
#     """Get AzureChatOpenAI instance"""
#     provider = configs_chat_llm.provider
#     match provider:
#         case "openai":
#             return get_openai_chat_llm(openai_api_key, configs_chat_llm)

#         case "azure":
#             return get_azure_chat_llm(configs_chat_llm)

#         case "huggingface":
#             raise NotImplementedError("HuggingFace chat LLM is not implemented yet.")

#         case _:
#             raise ValueError(f"Invalid configs.llm.provider: {provider}")


def get_llm(openai_api_key: str, configs_chat_openai: dict = CONFIGS_LLM.chat_openai):
    """Get llm instance"""
    key = (openai_api_key, LLM_KEY)
    if not SINGLETON_MANAGER.has_instance(key):
        if openai_api_key == "azure":
            llm = get_azure_chat_openai(configs_chat_openai)
        elif openai_api_key:
            llm = get_chat_openai(openai_api_key, configs_chat_openai)
        else:
            raise ValueError(f"Invalid openai_api_key: {openai_api_key}")

        # Validate openai_api_key
        llm.invoke("")
        SINGLETON_MANAGER.set_instance(key, llm)
    return SINGLETON_MANAGER.get_instance(key)


def get_chat_openai(
    openai_api_key: str, configs_chat_openai: dict = CONFIGS_LLM.chat_openai
) -> ChatOpenAI:
    """Get ChatOpenAI instance"""
    return ChatOpenAI(
        openai_api_key=openai_api_key,
        model=configs_chat_openai.model,
        temperature=configs_chat_openai.temperature,
    )


def get_azure_chat_openai(
    configs_chat_openai: dict = CONFIGS_LLM.chat_openai,
) -> AzureChatOpenAI:
    """Get AzureChatOpenAI instance"""
    return AzureChatOpenAI(
        # azure_endpoint=env["AZURE_OPENAI_ENDPOINT"],
        deployment_name=env["AZURE_OPENAI_LLM_DEPLOYMENT_NAME"],
        # openai_api_version=env["AZURE_OPENAI_API_VERSION"],
        # openai_api_key=env["AZURE_OPENAI_API_KEY"],
        model=env["AZURE_OPENAI_LLM_MODEL"],
        temperature=configs_chat_openai.temperature,
    )


############################################################
# Embeddings
############################################################
def get_embeddings(
    openai_api_key: str, configs_emb: dict = CONFIGS_LLM.openai_embeddings
):
    """Get embeddings instance"""
    key = (openai_api_key, EMBEDDINGS_KEY)
    if not SINGLETON_MANAGER.has_instance(key):
        if openai_api_key == "azure":
            emb = get_azure_embeddings(configs_emb)
        elif openai_api_key:
            emb = get_openai_embeddings(openai_api_key, configs_emb)
        else:
            raise ValueError(f"Invalid openai_api_key: {openai_api_key}")

        # Validate openai_api_key
        emb.embed_query("")
        SINGLETON_MANAGER.set_instance(key, emb)
    return SINGLETON_MANAGER.get_instance(key)


def get_openai_embeddings(
    openai_api_key: str, configs_emb: dict = CONFIGS_LLM.openai_embeddings
) -> OpenAIEmbeddings:
    """Get OpenAIEmbeddings instance"""
    return OpenAIEmbeddings(
        openai_api_key=openai_api_key,
        model=configs_emb.model,
    )


def get_azure_embeddings(
    configs_emb: dict = CONFIGS_LLM.openai_embeddings,
) -> AzureOpenAIEmbeddings:
    """Get AzureEmbeddings instance"""
    return AzureOpenAIEmbeddings(
        # azure_endpoint=env["AZURE_OPENAI_ENDPOINT"],
        deployment=env["AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT"],
        # openai_api_version=env["AZURE_OPENAI_API_VERSION"],
        # openai_api_key=env["AZURE_OPENAI_API_KEY"],
        model=env["AZURE_OPENAI_EMBEDDINGS_MODEL"],
    )


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
# CHAT_LLM = get_chat_llm()
# EMBEDDINGS = get_embeddings()
