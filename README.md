# 🧬 BioResearch Agent

基于RAG技术的生物医学论文智能分析系统

## 项目简介

上传任意生物医学论文PDF，自动完成：
- 📋 论文摘要生成
- 💡 创新点提取
- 🔬 研究方法分析
- 💬 自由问答

## 技术架构
## 技术栈

- LangChain 0.3
- ChromaDB（向量数据库）
- Ollama（本地LLM）
- Streamlit（前端界面）
- Python 3.9

## 本地运行

```bash
# 安装依赖
pip install langchain langchain-community langchain-ollama
pip install chromadb pypdf streamlit

# 下载模型
ollama pull llama3.2
ollama pull nomic-embed-text

# 启动
streamlit run app.py
```

## 应用场景

适用于科研人员、药企研发、医疗AI团队快速解析文献，
节省人工阅读时间，提升文献综述效率。
