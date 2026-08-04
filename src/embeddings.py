from sentence_transformers import SentenceTransformer
import chromadb

def load_embedding_model():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model

def embed_chunks(chunks, model):
    embeddings = model.encode(chunks, show_progress_bar=True)
    return embeddings

def get_chroma_collection(persist_directory="data/processed/chroma_db", collection_name="finsight_chunks"):
    client = chromadb.PersistentClient(path=persist_directory)
    collection = client.get_or_create_collection(name=collection_name)
    return collection

def store_chunks_in_chroma(chunks, embeddings, collection):
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    collection.add(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=chunks
    )

def query_collection(query_text, collection, model, n_results=5):
    query_embedding = model.encode([query_text])
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=n_results
    )
    return results

if __name__ == "__main__":
    from ingestion import extract_text_from_pdf, chunk_text

    model = load_embedding_model()
    collection = get_chroma_collection()

    if collection.count() == 0:
        text = extract_text_from_pdf("data/raw/infosys_annual_report_fy25.pdf")
        chunks = chunk_text(text)
        print("Total chunks to embed:", len(chunks))
        embeddings = embed_chunks(chunks, model)
        store_chunks_in_chroma(chunks, embeddings, collection)
        print("Stored", collection.count(), "chunks in ChromaDB.")
    else:
        print("Collection already has", collection.count(), "chunks. Skipping re-embedding.")

    query = "revenue growth year over year percentage"
    results = query_collection(query, collection, model, n_results=5)

    print("\n--- Query:", query, "---")
    for i, doc in enumerate(results["documents"][0]):
        print(f"\nResult {i+1}:")
        print(doc[:300])