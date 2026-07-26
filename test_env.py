from dotenv import load_dotenv
import os

load_dotenv()

print("GROQ_API_KEY loaded:", os.getenv("GROQ_API_KEY") is not None)
print("COHERE_API_KEY loaded:", os.getenv("COHERE_API_KEY") is not None)
print("TAVILY_API_KEY loaded:", os.getenv("TAVILY_API_KEY") is not None)