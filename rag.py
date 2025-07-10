import os
from io import BytesIO
from PIL import Image

import yaml
from boto3 import Session
import logging

from core.data_models import LLMResponse
from core.orchestrator import Orchestrator
from core.utils import from_list_to_messages


logger = logging.getLogger("app")

with open(os.getenv("API_SETTINGS_PATH")) as stream:
    api_config = yaml.safe_load(stream)
with open(os.getenv("CORE_SETTINGS_PATH")) as stream:
    rag_config = yaml.safe_load(stream)

logger.info("Initializing RAG model...")
RAG = Orchestrator(session=Session(), vector_store=api_config.get("vector-db-path"))

def update_rag(session: Session):
    global RAG
    logger.info("Updating RAG model...")
    RAG = Orchestrator(session=session, vector_store=api_config.get("vector-db-path"))
    logger.info("RAG updated")

def rag_schema():
    global RAG
    return Image.open(BytesIO(RAG.get_image()))

def rag_invoke(query,
               query_aug=False,
               retrieve_only=False,
               use_graph=False,
               use_embeddings=True,
               additional_context="",
               reranker="RRF",
               pre_translate=True,
               max_refs = 10,
               check_consistency=False,
               history=[],
               input_tokens_count=0,
               output_tokens_count=0) -> LLMResponse:
    if len(query)==0:
        return None
    else:
        response: LLMResponse = RAG.invoke({"query": query,
                               "history": from_list_to_messages(history),
                               "additional_context": additional_context,
                               "input_tokens_count": input_tokens_count,
                               "output_tokens_count": output_tokens_count,
                               "query_aug": query_aug,
                               "retrieve_only": retrieve_only,
                               "use_graph": use_graph,
                               "reranker": reranker,
                               "pre_translate": pre_translate,
                                            "check_consistency": check_consistency,
                               "max_refs": max_refs,
                               "use_embeddings": use_embeddings,
                               "pre_translate": True})
        return response