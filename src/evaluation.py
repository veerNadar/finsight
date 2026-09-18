import sys
import asyncio
sys.path.append("src")

from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.embeddings import HuggingFaceEmbeddings
from ragas.metrics.collections import Faithfulness, AnswerRelevancy

from agent import run_agent_query
from embeddings import load_embedding_model, get_chroma_collection, query_collection
from retrieval import get_cohere_client, rerank_results

import os
from dotenv import load_dotenv
load_dotenv()

embed_model = load_embedding_model()
chroma_collection = get_chroma_collection()
cohere_client = get_cohere_client()

# RAGAS needs its own LLM to act as a "judge" - we point it at Groq using an
# OpenAI-compatible client, since Groq's API is OpenAI-compatible.
judge_client = AsyncOpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)
judge_llm = llm_factory("openai/gpt-oss-120b", client=judge_client)

# AnswerRelevancy needs its own embeddings model. Using RAGAS's modern
# HuggingFaceEmbeddings class (capital F), which wraps sentence-transformers
# locally - same underlying library we already use for retrieval.
judge_embeddings = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")


def get_retrieved_contexts(query, n_results=10, top_n=2):
    """Get the actual chunks retrieved for a query - same logic as the agent's search_documents tool,
    but exposed here so we can capture the contexts for RAGAS evaluation."""
    results = query_collection(query, chroma_collection, embed_model, n_results=n_results)
    documents = results["documents"][0]
    reranked = rerank_results(query, documents, cohere_client, top_n=top_n)
    return [documents[r.index] for r in reranked]


test_cases = [
    {
        "question": "What was Infosys's operating margin in FY2024-25?",
        "ground_truth": "Infosys's operating margin for FY2024-25 was 21.1%."
    },
    {
        "question": "How many employees does Infosys have?",
        "ground_truth": "Infosys had over 3,20,000 employees as of FY2024-25, up from just over 3,17,000 at the end of FY2023-24."
    },
    {
        "question": "What was Infosys's revenue growth in FY2024-25?",
        "ground_truth": "Infosys reported 6.1% revenue growth (reported) and 4.2% constant currency growth for FY2024-25."
    },
]


async def evaluate_case(case, faithfulness_metric, relevancy_metric):
    answer = run_agent_query(case["question"])
    contexts = get_retrieved_contexts(case["question"])

    faithfulness_result = await faithfulness_metric.ascore(
        user_input=case["question"],
        response=answer,
        retrieved_contexts=contexts
    )

    relevancy_result = await relevancy_metric.ascore(
        user_input=case["question"],
        response=answer
    )

    return {
        "question": case["question"],
        "answer": answer,
        "ground_truth": case["ground_truth"],
        "faithfulness": faithfulness_result.value,
        "answer_relevancy": relevancy_result.value,
    }


async def main():
    faithfulness_metric = Faithfulness(llm=judge_llm)
    relevancy_metric = AnswerRelevancy(llm=judge_llm, embeddings=judge_embeddings)

    for case in test_cases:
        result = await evaluate_case(case, faithfulness_metric, relevancy_metric)
        print(f"\n{'='*60}")
        print(f"Question: {result['question']}")
        print(f"Answer: {result['answer']}")
        print(f"Ground truth: {result['ground_truth']}")
        print(f"Faithfulness score: {result['faithfulness']:.2f}")
        print(f"Answer relevancy score: {result['answer_relevancy']:.2f}")


if __name__ == "__main__":
    asyncio.run(main())