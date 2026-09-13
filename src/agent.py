import os
import sys
sys.path.append("src")

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain.agents import create_agent
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from groq import RateLimitError

from embeddings import load_embedding_model, get_chroma_collection, query_collection
from retrieval import get_cohere_client, rerank_results
from sentiment import load_finbert_model, analyze_sentiment
from tavily import TavilyClient

load_dotenv()

embed_model = load_embedding_model()
chroma_collection = get_chroma_collection()
cohere_client = get_cohere_client()
finbert_tokenizer, finbert_model = load_finbert_model()
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


@tool
def search_documents(query: str) -> str:
    """Search the Infosys annual report for relevant information using semantic search and reranking.
    Call this ONLY ONCE per distinct topic. Use this tool to answer questions about Infosys's financials,
    business segments, strategy, or operations based on their official annual report."""
    results = query_collection(query, chroma_collection, embed_model, n_results=10)
    documents = results["documents"][0]

    reranked = rerank_results(query, documents, cohere_client, top_n=2)

    output = ""
    for i, result in enumerate(reranked):
        doc_text = documents[result.index][:400]
        output += f"\n[Result {i+1}]\n{doc_text}\n"

    return output


@tool
def analyze_text_sentiment(text: str) -> str:
    """Analyze the financial sentiment (positive/negative/neutral) of a given piece of text.
    Use this after retrieving text to understand the tone of financial statements or commentary."""
    label, scores = analyze_sentiment(text, finbert_tokenizer, finbert_model)
    return f"Sentiment: {label} (positive: {scores['positive']}, negative: {scores['negative']}, neutral: {scores['neutral']})"


@tool
def web_search(query: str) -> str:
    """Search the live web for current information not available in the annual report,
    such as recent news, stock price, or events after the report's publication date."""
    response = tavily_client.search(query=query)
    output = ""
    for result in response["results"][:2]:
        output += f"\n{result['title']}\n{result['content'][:200]}\n"
    return output


SYSTEM_PROMPT = """You are a financial research assistant with access to exactly three tools:
search_documents, analyze_text_sentiment, and web_search. Only call these exact tools, never invent
other tools. Call search_documents AT MOST ONCE per question unless the user asks about a completely
different topic. After getting search results, use analyze_text_sentiment on that text if sentiment is
relevant, then immediately give your final answer. Do not repeat the same tool call. Be concise."""

llm = ChatGroq(model="openai/gpt-oss-120b", api_key=os.getenv("GROQ_API_KEY"))

agent = create_agent(
    llm,
    tools=[search_documents, analyze_text_sentiment, web_search],
    system_prompt=SYSTEM_PROMPT
)


@retry(
    retry=retry_if_exception_type(RateLimitError),
    wait=wait_exponential(multiplier=1, min=2, max=20),
    stop=stop_after_attempt(5)
)
def _invoke_agent_with_retry(question):
    return agent.invoke(
        {"messages": [("user", question)]},
        config={"recursion_limit": 10}
    )


def run_agent_query(question: str) -> str:
    """
    Public entry point: takes a plain-text question, runs it through the agent,
    and returns just the final answer text. This is what the Streamlit frontend
    will call later.
    """
    response = _invoke_agent_with_retry(question)
    final_message = response["messages"][-1]
    return final_message.content


if __name__ == "__main__":
    print("=== FinSight Agent (type 'quit' to exit) ===\n")

    while True:
        question = input("Ask a question about Infosys: ")
        if question.strip().lower() in ("quit", "exit"):
            break

        answer = run_agent_query(question)
        print(f"\n{answer}\n")
        print("-" * 60)