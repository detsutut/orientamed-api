from pydantic import BaseModel
from typing import List, Optional, Literal


class Concept(BaseModel):
    name: str
    id: str
    match_score: float
    semantic_tags: List[str] = []
    inconsistent: Optional[bool] = False

class Concepts(BaseModel):
    query: List[Concept] = []
    answer: List[Concept] = []

class RetrievedDocument(BaseModel):
    id: str
    page_content: str
    metadata: dict = {}
    score: Optional[float]

class RerankedDocument(BaseModel):
    id: str
    score: float

class References(BaseModel):
    embeddings: List[RetrievedDocument] = []
    graphs: List[RetrievedDocument] = []
    reranked: List[RerankedDocument] = []
    used: int

class ConsumedTokens(BaseModel):
    input: int = 0
    output: int = 0

class LLMResponseStatus(BaseModel):
    status: Literal['OK','ERROR','WARNING']
    details: Optional[str] = None

class LLMResponse(BaseModel):
    answer: Optional[str]
    consumed_tokens: ConsumedTokens
    references: References
    concepts: Concepts
    status: LLMResponseStatus