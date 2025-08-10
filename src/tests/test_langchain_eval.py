import pytest
from src.clients.llm_ollama import ask

def test_langchain_ollama_hello():
    result = ask("Say 'hello' in one short sentence.")
    assert "hello" in result.lower()
