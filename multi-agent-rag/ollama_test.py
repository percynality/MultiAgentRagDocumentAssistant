
from langchain_ollama import ChatOllama

# Connect Python to our local Ollama model
llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)

# Send a test question
response = llm.invoke(
    "Explain what Retrieval-Augmented Generation (RAG) means in simple words."
)

# Display the answer
print("\n--- LOCAL AI RESPONSE ---")
print(response.content)
