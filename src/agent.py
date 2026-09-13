import os
import sys
sys.path.append("src")

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain.agents import create_agent
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from groq import RateLimitError, BadRequestError
from langgraph.errors import GraphRecursionError

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
    """Search the Infosys FY2024-25 annual report for relevant information using semantic search
    and reranking. This document contains FY2024-25 figures and one prior-year (FY2023-24) comparison
    only - it does NOT contain data for years before FY2023-24. Use this tool for questions about
    Infosys's recent financials, business segments, strategy, or operations."""
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
    """Search the live web for current information not available in the annual report -
    such as recent news, current stock price, historical data older than FY2023-24, or events
    after the report's publication date. Use this as a fallback when search_documents doesn't
    have what's needed."""
    response = tavily_client.search(query=query)
    output = ""
    for result in response["results"][:2]:
        output += f"\n{result['title']}\n{result['content'][:200]}\n"
    return output


SYSTEM_PROMPT = """You are a financial research assistant with access to exactly three tools:
search_documents, analyze_text_sentiment, and web_search. Only call these exact tools, never invent
other tools.

Tool-calling rules:
- NEVER call the same tool with the same or near-identical query more than once.
- For multi-part questions, you may call a tool multiple times with genuinely different queries,
  but limit yourself to AT MOST 3 tool calls total per question.
- If a question requires more than 3 distinct pieces of information (e.g. 4+ years of data),
  do NOT attempt it in full. Instead, answer for as many parts as you reasonably can within 3 tool
  calls, and tell the user to ask about fewer items at a time for the rest.
- search_documents only covers FY2024-25 and FY2023-24. For anything outside that range, or for
  current/live information, use web_search instead.
- Use analyze_text_sentiment when the tone of financial commentary is relevant.
- NEVER state a specific number, statistic, or fact unless it appears explicitly in a tool's output.
  If a tool doesn't provide data for part of the question, say so directly instead of guessing or
  estimating.
- When reporting financial figures, always double-check that the currency unit you state (₹ crore,
  $ billion, etc.) actually matches the number you are reporting - do not mix a numeric value from
  one currency with a unit label from another.
- Be concise in your final answer."""

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
        config={"recursion_limit": 15}
    )


def run_agent_query(question: str) -> str:
    """
    Public entry point: takes a plain-text question, runs it through the agent,
    and returns just the final answer text. This is what the Streamlit frontend
    will call later.
    """
    try:
        response = _invoke_agent_with_retry(question)
        final_message = response["messages"][-1]
        return final_message.content
    except GraphRecursionError:
        return ("I wasn't able to fully answer this within my step limit. This usually happens "
                "with questions needing many distinct facts at once - try asking about fewer "
                "items (e.g. one or two years/metrics) per question.")
    except RateLimitError:
        return "I'm currently rate-limited by the LLM provider. Please wait a moment and try again."
    except BadRequestError:
        return ("I ran into trouble processing this question, likely because it required too many "
                "steps to research fully. Please try breaking it into smaller, more specific "
                "questions (e.g. one year or metric at a time).")


if __name__ == "__main__":
    print("=== FinSight Agent (type 'quit' to exit) ===\n")

    while True:
        question = input("Ask a question about Infosys: ")
        if question.strip().lower() in ("quit", "exit"):
            break

        answer = run_agent_query(question)
        print(f"\n{answer}\n")
        print("-" * 60)