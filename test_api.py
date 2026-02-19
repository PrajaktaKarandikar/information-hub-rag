# test_api.py - Test the API.

import json
import requests

BASE_URL = "http://localhost:8000"

def test_health():
    response = requests.get(f"{BASE_URL}/health")
    print("Health Check Response:", response.json())

def test_query(question: str):
    response = requests.post(
        f"{BASE_URL}/query", 
        json={"question": question, "return_sources": True}
    )
    result = response.json()
    print(f"\nQuestion: {question}")
    print(f"Answer: {result['answer'][200:]}...") # Print only the first 200 characters of the answer for brevity
    print(f"Sources: {len(result.get('sources', []))}") # Print the number of sources returned
    print(f"Tokens Used: {result.get('tokens_used', 'N/A')}") # Print tokens used if available

def test_ingest():
    # Example document to ingest
    sources = [
        "https://en.wikipedia.org/wiki/Artificial_intelligence"
    ]
    response = requests.post(
        f"{BASE_URL}/ingest", 
        json={"sources": sources, "replace_existing": True}
    )
    print("Ingest Response:", response.json())

if __name__ == "__main__":
    test_health()
    test_ingest()
    test_query("What is artificial intelligence?")
    test_query("LLMs are transforming the way we interact with technology. Can you explain how they work?")