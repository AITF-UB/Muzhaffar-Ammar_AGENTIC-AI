import sys
import os

# Add beta-agentic to path so we can import tools
sys.path.insert(0, r"c:\Users\Ammar\Projek\agentic-ai\beta-agentic")
from tools import RAGEngine

query = "perangkat keras"
print(f"Querying: {query}")
result = RAGEngine.unified_search(query, "bacaan")
print("Number of texts:", len(result["text"]))
print("Images extracted:", result["images"])
