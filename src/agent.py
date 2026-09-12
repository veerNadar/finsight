import os
from dotenv import load_dotenv
from groq import Groq
from tavily import TavilyClient

load_dotenv()

def test_groq():
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "user", "content": "In one sentence, what is Infosys?"}
        ]
    )
    return response.choices[0].message.content

def test_tavily():
    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    response = client.search(query="Infosys latest quarterly results 2026")
    return response

if __name__ == "__main__":
    print("=== Testing Groq ===")
    groq_answer = test_groq()
    print(groq_answer)

    print("\n\n=== Testing Tavily ===")
    tavily_results = test_tavily()
    for result in tavily_results["results"][:3]:
        print(f"\nTitle: {result['title']}")
        print(f"URL: {result['url']}")
        print(f"Snippet: {result['content'][:200]}")