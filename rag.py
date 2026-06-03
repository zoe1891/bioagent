from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM

# 读取PDF
def read_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# 切块
def split_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    return splitter.split_text(text)

# 加载已有向量库
def load_vectorstore():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
    return vectorstore

# 问答
def ask(question, vectorstore):
    # 检索最相关的3段内容
    docs = vectorstore.similarity_search(question, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # 构建prompt
    prompt = f"""你是一个科研论文分析助手。根据以下论文内容回答问题。

论文内容：
{context}

问题：{question}

请用中文回答："""
    
    llm = OllamaLLM(model="llama3.2")
    response = llm.invoke(prompt)
    return response

# 主程序
print("加载向量库...")
vectorstore = load_vectorstore()
print("加载完成！开始问答（输入q退出）\n")

while True:
    question = input("你的问题：")
    if question == "q":
        break
    print("\n回答中...\n")
    answer = ask(question, vectorstore)
    print(f"回答：{answer}\n")
    print("-" * 50 + "\n")
