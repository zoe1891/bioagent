from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="llama3.2")
response = llm.invoke("用中文介绍一下什么是RAG技术")
print(response)
