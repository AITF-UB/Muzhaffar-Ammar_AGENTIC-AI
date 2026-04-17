from qdrant_client import QdrantClient

client = QdrantClient(host="localhost", port=6333)
collection_name = "pdf_rag_collection_v2"

try:
    response = client.scroll(
        collection_name=collection_name,
        limit=2,
        with_payload=True
    )
    for point in response[0]:
        print(f"Point ID: {point.id}")
        print(f"Payload: {point.payload}")
        print("-" * 50)
except Exception as e:
    print(f"Error: {e}")
