from pydantic import BaseModel
from typing import List, Optional

class Concept(BaseModel):
    name: str
    id: str
    match_score: float
    semantic_tags: List[str]

class Concepts(BaseModel):
    query: List[Concept]
    answer: List[Concept]

class RetrievedDocument(BaseModel):
    id: str
    page_content: str
    metadata: dict
    score: Optional[float]

class RetrievedDocuments(BaseModel):
    embeddings: List[RetrievedDocument]
    graphs: List[RetrievedDocument]
    reranked: List[str]

class ConsumedTokens(BaseModel):
    input: int
    output: int

class LLMResponse(BaseModel):
    answer: str
    consumed_tokens: ConsumedTokens
    retrieved_documents: RetrievedDocuments
    concepts: Concepts