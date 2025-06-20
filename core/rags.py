from typing import Any, Literal
from boto3 import Session
from typing_extensions import List, TypedDict
import textwrap
import json
import requests
import pandas as pd
import logging
import numpy as np

from langchain_aws import InMemoryVectorStore
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.types import Command
from langchain_core.messages.human import HumanMessage
import os
import yaml

from core.kg_retriever import shortest_path_bewteen, shortest_path_id, get_chunk
from core.languagemodel import LanguageModel
from core.retriever import Retriever
from core.data_models import RetrievedDocument

logger = logging.getLogger('app.'+__name__)
logging.getLogger("langchain_aws").setLevel(logging.WARNING)
logging.getLogger("langchain_core").setLevel(logging.WARNING)

with open(os.getenv("CORE_SETTINGS_PATH")) as stream:
    rag_config = yaml.safe_load(stream)

def messages_to_history_str(messages: list[BaseMessage]) -> str:
    """Convert messages to a history string."""
    string_messages = []
    for message in messages:
        role = message.type
        content = message.content
        string_message = f"{role}: {content}"

        additional_kwargs = message.additional_kwargs
        if additional_kwargs:
            string_message += f"\n{additional_kwargs}"
        string_messages.append(string_message)
    return "\n".join(string_messages)


class Prompts:
    def __init__(self, jsonfile: str):
        with open(jsonfile, 'r') as file:
            data = json.load(file)
            for k, v in data.items():
                self.__setattr__(k, self.__parse__(v))

    def __parse__(self, prompt_dicts: List[dict]):
        return ChatPromptTemplate([(d["role"], d["content"]) for d in prompt_dicts])


# Define state for application
from typing_extensions import Annotated
from operator import add
class State(TypedDict):
    #INPUTS
    query: str # the user query
    history: List[BaseMessage] # all the interactions between ai and user
    additional_context: str # additional info added by the user to be considered a valid source
    query_aug: bool # use or not query augmentation technique before passing the query to the retriever
    use_graph: bool
    use_embeddings: bool
    retrieve_only: bool
    pre_translate: bool
    #INTERNAL
    answer_generated: bool
    #OUTPUTS
    input_tokens_count: Annotated[int, add] # amount of input tokens processed by the whole chain of llm calls triggered in this round
    output_tokens_count: Annotated[int, add] # amount of output tokens processed by the whole chain of llm calls triggered in this round
    answer: str # textual answer generated by the system and returned to the user
    query_concepts: list[dict] # list of concepts extracted from input query
    answer_concepts: list[dict]  # list of concepts extracted from generated answer
    docs_graph: list[RetrievedDocument] # retrieved documents to use as context source
    docs_embeddings: list[RetrievedDocument] # retrieved documents to use as context source
    status: str # OK or NO_RETRIEVE or NOT_ALLOWED

class Rag:

    def __init__(self, session: Session,
                 vector_store: InMemoryVectorStore | str | None = None):
        client = session.client("bedrock-runtime", region_name=rag_config.get("bedrock").get("region"))
        self.prompts = Prompts(rag_config.get("promptfile"))
        self.llm = LanguageModel(client=client,
                                 model = rag_config.get("bedrock").get("models").get("model-id"),
                                 model_low = rag_config.get("bedrock").get("models").get("low-model-id", None),
                                 model_pro = rag_config.get("bedrock").get("models").get(" pro-model-id", None))
        self.retriever = Retriever(client=client,
                                   embedder=rag_config.get("bedrock").get("embedder-id"),
                                   vector_store=vector_store)

        graph_builder = StateGraph(state_schema=State)
        graph_builder.set_entry_point("orchestrator")
        graph_builder.add_node("orchestrator", self.orchestrator)
        graph_builder.add_node("history_consolidator", self.history_consolidator)
        graph_builder.add_node("augmentator", self.augmentator)
        graph_builder.add_node("emb_retriever", self.emb_retriever)
        graph_builder.add_node("ans_generator", self.ans_generator)
        graph_builder.add_node("consistency_checker", self.consistency_checker)
        graph_builder.add_node("concept_extractor", self.concept_extractor)
        graph_builder.add_node("kg_retriever", self.kg_retriever)
        self.graph = graph_builder.compile()

    def orchestrator(self, state: State) -> Command[Literal["augmentator", "history_consolidator"]]:
        logger.info(f"Dispatching request...")
        logger.debug(f"Request details:\n\t-Query: {textwrap.shorten(state['query'],30)}\n\t-History: {len(state['history'])} messages\n\t-Additional context: {state['additional_context']}\n\t-Query augmentation: {state['query_aug']}\n\t-Graph DB: {state['use_graph']}\n\t-Vector DB: {state['use_embeddings']}\n\t-Retrieve only: {state['retrieve_only']}\n\t-Pre-translate: {state['pre_translate']}")
        previous_user_interactions = [message for message in state["history"] if type(message) is HumanMessage]
        if len(previous_user_interactions) > 0:
            return Command(update={"answer_generated":False},
                           goto="history_consolidator")
        else:
            return Command(update={"answer_generated":False},
                           goto="augmentator")

    def history_consolidator(self, state: State) -> Command[Literal["orchestrator"]]:
        logger.info(f"Consolidating history...")
        messages = self.prompts.history_consolidation.invoke({"question": state["query"],
                                                              "history": messages_to_history_str(
                                                                  state["history"])}).messages
        response = self.llm.generate(messages=messages)
        consolidated_query = response.content
        logger.info(f"Consolidated query: {textwrap.shorten(consolidated_query, width=30)}")
        return Command(
            update={"query": consolidated_query,
                    "history": [],
                    "input_tokens_count": response.usage_metadata["input_tokens"],
                    "output_tokens_count": response.usage_metadata["output_tokens"]},
            goto="orchestrator",
        )

    def augmentator(self, state: State) -> Command[Literal["emb_retriever","ans_generator"]]:
        update = {}
        if state["query_aug"]:
            logger.info(f"Expanding Query...")
            messages = self.prompts.query_expansion.invoke({"question": state["query"]}).messages
            response = self.llm.generate(messages=messages)
            augmented_query = response.content
            logger.info(f"Expanded query: {textwrap.shorten(augmented_query, width=30)}")
            update = {"query": augmented_query,
                        "input_tokens_count": response.usage_metadata["input_tokens"],
                        "output_tokens_count": response.usage_metadata["output_tokens"]
                        }
        if state["use_embeddings"] or state["use_graph"]:
            return Command(update=update, goto="emb_retriever")
        else:
            return Command(update=update, goto="ans_generator")

    def concept_extractor(self, state: State) -> Command[Literal["kg_retriever","consistency_checker"]]:
        if not state["use_graph"]:
            logger.debug(f"Graph not activate, bypassing concept extraction...")
            if state["answer_generated"]:
                return Command(update={"status": "OK"},goto=END)
            else:
                return Command(goto="ans_generator")
        if state["answer_generated"]:
            logger.debug(f"Running concept extraction on answer")
        else:
            logger.debug(f"Running concept extraction on query")
        if state["answer_generated"] and state["retrieve_only"]:
            logger.debug(f"Retrieve only, bypassing concept extraction for answer...")
            return Command(update={"status": "OK"},goto=END)
        input_tokens = 0
        output_tokens = 0
        if state["pre_translate"]:
            logger.debug(f"Translating Text in English before feeding to Concept Extractor...")
            messages = self.prompts.translation.invoke({"source_lang": "Italian",
                                                          "target_lang": "English",
                                                          "source_text":  state["query"] if not state["answer_generated"] else state["answer"],
                                                          }).messages
            response = self.llm.generate(messages=messages, level="pro")
            input_tokens = response.usage_metadata["input_tokens"]
            output_tokens = response.usage_metadata["output_tokens"]
            text_to_send = response.content
            logger.debug(f"Translated test: {textwrap.shorten(text_to_send, width=30)}")
        else:
            text_to_send = state["query"] if not state["answer_generated"] else state["answer"]
            logger.debug(f"Translated test: {textwrap.shorten(text_to_send, width=30)}")
        logger.info(f"Extracting Concepts...")
        url = "https://dheal-com.unipv.it:7878/extract"
        params = {'text': text_to_send, 'o': 100, 'p': False}
        response = requests.get(url, params=params)
        if response.status_code != 200:
            logger.error("Error during concept extraction. Further investigation needed.")
            logger.debug(f"Raw response: {response}")
            concepts = []
        else:
            try:
                concepts = pd.DataFrame(response.json()).to_dict(orient='records')
            except Exception as e:
                logger.error(f"Error during concept extraction: {e}")
                logger.debug(f"Raw response: {response}")
                concepts = []
        if len(concepts)==0:
            logger.debug("No concepts found, trying with premium translation")
            params = {'text': text_to_send, 'o': 100, 'p': True}
            response = requests.get(url, params=params)
            concepts = pd.DataFrame(response.json()).to_dict(orient='records')
        if not state["answer_generated"]:
            return Command(
                update={"query_concepts": concepts,
                        "input_tokens_count": input_tokens,
                        "output_tokens_count": output_tokens
                        },
                goto="kg_retriever",
            )
        else:
            return Command(
                update={"answer_concepts": concepts,
                        "input_tokens_count": input_tokens,
                        "output_tokens_count": output_tokens
                        },
                goto="consistency_checker",
            )

    def emb_retriever(self, state: State) -> Command[Literal["concept_extractor", END]]:
        if not state["use_embeddings"]:
            return Command(goto="concept_extractor")
        logger.info(f"Retrieving Documents...")
        retrieved_docs = self.retriever.retrieve_with_scores(state["query"], n=10, score_threshold=0.4)
        logger.info(f"{len(retrieved_docs)} documents retrieved.")
        additional_context = state.get("additional_context", None)
        if len(retrieved_docs) == 0 and (type(additional_context) is not str or additional_context == ""):
            return Command(
                update={"docs_embeddings": retrieved_docs,
                        "answer": "",
                        "status": "NO_RETRIEVE"},
                goto=END,
            )
        else:
            return Command(
                update={"docs_embeddings": retrieved_docs},
                goto="concept_extractor",
            )

    def consistency_checker(self, state: State) -> Command[Literal[END]]:
        logger.info(f"Checking answer consistency...")
        qc = state["query_concepts"]
        qc_ids = [c["id"] for c in qc]
        qc_names = [c["name"] for c in qc]
        inconsistent_concepts = []
        answer = state["answer"]
        for answer_concept in state["answer_concepts"]:
            # For answer concepts that are not in the query concepts, check if there is at least one path between them and a query concept. If not, add them to the list of inconsistent concepts.
            if answer_concept["id"] not in qc_ids and answer_concept["name"] not in qc_names:
                path_found = False
                for question_concept_id in qc_ids:
                    if shortest_path_bewteen(id1=answer_concept["id"], id2=question_concept_id, max_hops=rag_config.get("graph").get("max_hops",5)):
                        path_found = True
                        break
                if not path_found:
                    inconsistent_concepts.append(answer_concept["name"]+" ("+answer_concept["id"]+")")
        if inconsistent_concepts:
            ic_string = ', '.join(inconsistent_concepts)
            answer_warning = f"<div style='padding-top:15px;'><div id='warning'>⚠️ Alcuni concetti menzionati nella risposta non sembrano essere collegati con quelli menzionati nella query. Valutare la risposta in modo scrupoloso e non utilizzare direttamente per prendere decisioni mediche. ({int((len(inconsistent_concepts)/len(state['answer_concepts']))*100)}%)</div></div>"
            logger.warning(f"INCONSISTENCY WARNING: {ic_string}")
            answer = answer+"\n"+answer_warning
        return Command(
            update={"answer": answer, "status": "OK"},
            goto=END,
        )

    def kg_retriever(self, state: State) -> Command[Literal["ans_generator"]]:
        logger.info(f"Retrieving Nodes...")
        paths = []
        for concept in state["query_concepts"]:
            paths.extend(shortest_path_id(id=concept["id"], max_hops=3))
        result = {}
        for path in paths:
            item_id = path["id"]
            if item_id not in result or path["nodeCount"] < result[item_id]["nodeCount"]:
                result[item_id] = path
        deduplicated_paths = list(result.values())
        deduplicated_sorted_paths = sorted(deduplicated_paths, key=lambda x: x["nodeCount"])
        retrieved_docs = []
        for path in deduplicated_sorted_paths:
            path_string = ""
            for element in path["path"]:
                if type(element) is dict:
                    path_string += "["+element.get("FSN","Chunk")+"]"
                elif type(element) is str:
                    path_string += "--["+element+"]--"
            chunk = get_chunk(path["id"])
            retrieved_docs.append(RetrievedDocument(id=chunk["chunkId"],
                                                    page_content=chunk["text"],
                                                    score= path["nodeCount"],
                                                    metadata={"title": chunk["title"],
                                                              "path": path_string,
                                                              "source": chunk["chunkId"].split("txt")[0]}))
        return Command(
            update={"docs_graph": retrieved_docs},
            goto="ans_generator",
        )

    def ans_generator(self, state: State) -> Command[Literal["concept_extractor"]]:
        if state["retrieve_only"]:
            return Command(update={"answer": "*Nessuna risposta generata. Le risposte sono disattivate*",
                                   "answer_generated": True,
                                   "input_tokens_count": 0,
                                   "output_tokens_count": 0},
                           goto="concept_extractor")
        logger.info(f"Generating...")
        logger.info(f"Organizing references for answer generation...")
        doc_strings = []
        already_used_docs = []
        for i, doc in enumerate(state.get("docs_embeddings",[])):
            doc_strings.append(f"Source {i+1}:\n\"{doc.page_content}\"")
            already_used_docs.append(doc.metadata.get("doc_id"))
        # ADD KG-RELATED CHUNKS, BUT ONLY IF NOT ALREADY RETRIEVED BY STANDARD RAG
        scores = [doc.score for doc in state.get("docs_graph",[])]
        if len(scores)>0:
            min_score_docs = [state.get("docs_graph")[i] for i in np.where(scores == np.min(scores))[0]]
            not_overlapping_docs = []
            for doc in min_score_docs:
                if doc.metadata.get("doc_id") not in already_used_docs:
                    not_overlapping_docs.append(doc)
            for i,doc in enumerate(not_overlapping_docs):
                doc_strings.append(f"Source KG{i+1}:\n\"{doc.page_content}\"")
        #ADDITIONAL CONTEXT
        additional_context = state.get("additional_context", None)
        if type(additional_context) is str and additional_context != "":
            logger.info(f"Appending additional context...")
            doc_strings.append(f"Source [0]:\n\"{additional_context}\"")
        #ANSWERING
        if len(doc_strings)>0:
            docs_content = "\n\n".join(doc_strings)
            messages = self.prompts.question_with_context_inline_cit.invoke({"question": state["query"], "context": docs_content}).messages
        else:
            messages = self.prompts.question_open.invoke({"question": state["query"]}).messages
        response = self.llm.generate(messages=messages, level="pro")
        return Command(update={"answer": response.content,
                               "answer_generated": True,
                               "input_tokens_count": response.usage_metadata["input_tokens"],
                               "output_tokens_count": response.usage_metadata["output_tokens"]},
                       goto="concept_extractor")

    def invoke(self, input: dict[str, Any]):
        return self.graph.invoke(input)

    def get_image(self):
        try:
            return self.graph.get_graph().draw_mermaid_png()
        except Exception as e:
            logger.error(f"Error drawing graph: {e}")
            return None
