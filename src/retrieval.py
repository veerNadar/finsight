import os
from dotenv import load_dotenv
import cohere

load_dotenv()

def get_cohere_client():
    api_key = os.getenv("COHERE_API_KEY")
    client = cohere.Client(api_key)
    return client

def rerank_results(query, documents, cohere_client, top_n=3):
    response = cohere_client.rerank(
        model="rerank-english-v3.0",
        query=query,
        documents=documents,
        top_n=top_n
    )
    return response.results

if __name__ == "__main__":
    import sys
    sys.path.append("src")
    from embeddings import load_embedding_model, get_chroma_collection, query_collection

    model = load_embedding_model()
    collection = get_chroma_collection()

    query = "revenue growth year over year percentage"
    results = query_collection(query, collection, model, n_results=5)
    documents = results["documents"][0]

    print("--- BEFORE reranking (ChromaDB order) ---")
    for i, doc in enumerate(documents):
        print(f"\nResult {i+1}:")
        print(doc[:200])

    cohere_client = get_cohere_client()
    reranked = rerank_results(query, documents, cohere_client, top_n=5)

    print("\n\n--- AFTER reranking (Cohere order) ---")
    for i, result in enumerate(reranked):
        original_index = result.index
        relevance_score = result.relevance_score
        print(f"\nRank {i+1} (was Result {original_index+1}, score: {relevance_score:.4f}):")
        print(documents[original_index][:200])