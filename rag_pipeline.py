from dotenv import load_dotenv
import os

# 加载 .env 文件
load_dotenv()

import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from openai import OpenAI



# 初始化 Embedding 模型和向量数据库
embeddings = DashScopeEmbeddings(model="text-embedding-v3") # 建议使用v3，效果更好
db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# 初始化千问大模型客户端
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

def add_documents_to_vectorstore(documents: list[str], metadatas: list[dict], ids: list[str]):
    """将处理好的文本块添加到向量数据库中"""
    if documents:
        db.add_texts(texts=documents, metadatas=metadatas, ids=ids)
        db.persist()

def generate_answer(question: str, top_k: int = 8) -> dict:
    """RAG核心流程：检索 + 生成"""
    # 🔥 关键修复：不再使用分数阈值，改用最基础的 top_k 检索，彻底杜绝“未找到”的问题
    retriever = db.as_retriever(search_kwargs={"k": top_k})
    docs = retriever.invoke(question)

    # 组装上下文
    if not docs:
        return {"answer": "知识库中未找到任何相关文档，建议您上传相关文档。", "sources": []}

    context = "\n\n---\n\n".join([doc.page_content for doc in docs])

    # 构建 Prompt，让 AI 自己判断资料是否充足
    prompt = f"""
你是一位专业的知识助手。请严格基于以下【参考资料】回答【用户问题】。

【参考资料】：
{context}

【用户问题】：
{question}

【回答要求】：
1. 仔细阅读参考资料，提取与问题相关的信息。
2. 如果资料中包含部分相关信息，请尽可能完整作答，并列出资料中提到的部分；如果完全没有相关内容，再提示“根据现有资料无法准确回答”。
3. 涉及对比（如标准版与专业版）时，尽量使用列表或表格形式清晰展示。
"""

    # 调用大模型生成最终答案
    response = client.chat.completions.create(
        model="qwen-plus", 
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2, 
    )
    
    answer = response.choices[0].message.content
    
    # 返回结果
    sources = [{"index": i + 1, "snippet": doc.page_content[:100] + "..."} for i, doc in enumerate(docs)]
    return {"answer": answer, "sources": sources}