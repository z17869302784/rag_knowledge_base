import uuid
from pydantic import BaseModel
from fastapi import Body
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from document_processor import extract_text, chunk_text
from rag_pipeline import generate_answer, add_documents_to_vectorstore

# 初始化 FastAPI 应用
app = FastAPI(title="企业知识库问答系统")
# 挂载静态文件目录，让外部可以通过 /static 路径访问
app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/")
def read_root():
    return {"message": "欢迎来到企业知识库问答系统"}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    上传文档：解析 -> 分块 -> 存入向量库
    """
    try:
        content = await file.read()
        filename = file.filename

        # 1. 解析文本
        text = extract_text(filename, content)
        if not text.strip():
            raise HTTPException(status_code=400, detail="文档内容为空")

        # 2. 文本分块
        chunks = chunk_text(text)
        
        # 3. 准备元数据和ID
        metadatas = [{"source": filename, "chunk_id": i} for i in range(len(chunks))]
        ids = [f"{filename}_{uuid.uuid4().hex[:8]}_{i}" for i in range(len(chunks))]

        # 4. 存入向量数据库
        add_documents_to_vectorstore(chunks, metadatas, ids)

        return {
            "status": "success",
            "filename": filename,
            "chunks_count": len(chunks),
            "message": "文档已成功添加到知识库！"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 定义一个专门用来接收提问的数据模型
class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
async def ask_question(question_data: QuestionRequest):
    """
    基于知识库的问答接口
    """
    question = question_data.question
    if not question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")

    # 获取 AI 的回答
    result = generate_answer(question)

    # 检查返回的类型，防止报错，并去掉开头的短横线
    if isinstance(result, dict) and "answer" in result:
        answer = result["answer"]
        if answer.startswith("- "):
            answer = answer[2:] # 去掉开头的 '- ' (2个字符)
        result["answer"] = answer # 把处理过的答案放回去
    
    return result

# 启动命令：uvicorn main:app --reload --port 8000