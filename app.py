import streamlit as st
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
import tempfile, os

st.title("🧬 BioResearch Agent")
st.caption("上传论文PDF，自动分析创新点、摘要，支持自由问答")

# 初始化session
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# 左侧上传区
with st.sidebar:
    st.header("上传论文")
    uploaded_file = st.file_uploader("选择PDF文件", type="pdf")
    
    if uploaded_file and st.session_state.vectorstore is None:
        with st.spinner("正在处理论文..."):
            # 读取PDF
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
                f.write(uploaded_file.read())
                tmp_path = f.name
            
            reader = PdfReader(tmp_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            
            # 切块向量化
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500, chunk_overlap=50)
            chunks = splitter.split_text(text)
            
            embeddings = OllamaEmbeddings(model="nomic-embed-text")
            st.session_state.vectorstore = Chroma.from_texts(
                texts=chunks,
                embedding=embeddings
            )
            os.unlink(tmp_path)
        st.success(f"处理完成！共{len(chunks)}个文本块")
    
    # 快捷按钮
    if st.session_state.vectorstore:
        st.header("快捷分析")
        if st.button("📋 生成摘要"):
            st.session_state.messages.append(
                {"role": "user", "content": "请总结这篇论文的主要内容"})
        if st.button("💡 提取创新点"):
            st.session_state.messages.append(
                {"role": "user", "content": "这篇论文的创新点是什么？"})
        if st.button("🔬 研究方法"):
            st.session_state.messages.append(
                {"role": "user", "content": "这篇论文使用了什么研究方法？"})

# 主界面问答
if st.session_state.vectorstore is None:
    st.info("👈 请先在左侧上传论文PDF")
else:
    # 显示历史消息
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    
    # 处理最新问题
    if st.session_state.messages and \
       st.session_state.messages[-1]["role"] == "user":
        question = st.session_state.messages[-1]["content"]
        
        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                docs = st.session_state.vectorstore.similarity_search(
                    question, k=3)
                context = "\n\n".join([d.page_content for d in docs])
                prompt = f"""你是科研论文分析助手。根据论文内容回答问题。

论文内容：
{context}

问题：{question}

请用中文详细回答："""
                llm = OllamaLLM(model="llama3.2")
                answer = llm.invoke(prompt)
                st.write(answer)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer})
    
    # 输入框
    if prompt := st.chat_input("输入你的问题..."):
        st.session_state.messages.append(
            {"role": "user", "content": prompt})
        st.rerun()
