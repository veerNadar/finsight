import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain.agents import create_agent

load_dotenv()

@tool
def add_numbers(a: int, b: int) -> int:
    """Add two numbers together and return the result. Always use this tool for addition, never calculate it yourself."""
    print(f"\n>>> TOOL CALLED: add_numbers({a}, {b})")
    return a + b

llm = ChatGroq(model="openai/gpt-oss-120b", api_key=os.getenv("GROQ_API_KEY"))

agent = create_agent(llm, tools=[add_numbers])

if __name__ == "__main__":
    response = agent.invoke({
        "messages": [("user", "What is 4837 plus 9265?")]
    })

    for message in response["messages"]:
        print(f"\n[{message.type}]")
        print(message.content)