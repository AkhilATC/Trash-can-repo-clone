from langchain_ollama import ChatOllama

# Connect to local Ollama
llm = ChatOllama(
    model="qwen3:0.6b",
    base_url="http://localhost:11434",
    temperature=0.7,
    num_ctx=2048,
    num_predict=256,
    reasoning=False
)