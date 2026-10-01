"""Result model of the ``search_banking_policy`` tool."""

from pydantic import BaseModel, ConfigDict


class PolicyPassage(BaseModel):
    model_config = ConfigDict(frozen=True)

    source: str  # e.g. "fees.md"
    title: str  # document title
    section: str  # section heading
    text: str
    score: float  # cosine similarity to the question


class PolicySearchResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    query: str
    passages: list[PolicyPassage]
    # False = the knowledge base does not cover the question; the agent must not answer from guesswork.
    sufficient: bool
    reason: str  # ok | empty_index | low_similarity | unknown_terms
    embedding_model: str
