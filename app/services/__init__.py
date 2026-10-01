"""Domain services: the banking business logic behind the MCP tools.

Services know nothing about MCP, HTTP or LLMs. They take a database session, the
authenticated ``Principal`` (for personal data) and validated arguments, and
return Pydantic result models. That keeps them easy to unit test and reusable
from the REST API as well as from MCP.
"""
