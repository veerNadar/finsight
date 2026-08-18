from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

def load_finbert_model():
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    return tokenizer, model

def analyze_sentiment(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    scores = torch.nn.functional.softmax(outputs.logits, dim=1)
    labels = ["positive", "negative", "neutral"]
    scores_dict = {labels[i]: round(scores[0][i].item(), 4) for i in range(3)}
    predicted_label = labels[scores.argmax().item()]
    return predicted_label, scores_dict

if __name__ == "__main__":
    import sys
    sys.path.append("src")
    from embeddings import load_embedding_model, get_chroma_collection, query_collection

    tokenizer, model = load_finbert_model()

    # Test on hand-written examples first
    test_sentences = [
        "Revenue grew 2.4% year-over-year, driven by strong performance in financial services.",
        "The company reported a significant decline in operating margin due to rising costs.",
        "The board approved the quarterly dividend as per standard schedule."
    ]

    print("=== Sanity check on hand-written examples ===")
    for sentence in test_sentences:
        label, scores = analyze_sentiment(sentence, tokenizer, model)
        print(f"\nText: {sentence}")
        print(f"Predicted sentiment: {label}")
        print(f"Scores: {scores}")

    # Now test on real retrieved chunks
    print("\n\n=== Sentiment on real document chunks ===")
    embed_model = load_embedding_model()
    collection = get_chroma_collection()

    query = "revenue growth year over year percentage"
    results = query_collection(query, collection, embed_model, n_results=3)
    documents = results["documents"][0]

    for doc in documents:
        label, scores = analyze_sentiment(doc, tokenizer, model)
        print(f"\nChunk: {doc[:200]}...")
        print(f"Predicted sentiment: {label}")
        print(f"Scores: {scores}")