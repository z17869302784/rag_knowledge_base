from langchain_text_splitters import RecursiveCharacterTextSplitter

def extract_text(filename: str, content: bytes) -> str:
    """解析上传的文档内容"""
    if filename.endswith(".txt"):
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            text = content.decode("gbk", errors="ignore")
        return text
    # 如果后续需要支持 pdf/docx，可以在这里扩展
    return ""

def chunk_text(text: str, chunk_size: int = 300, chunk_overlap: int = 100):
    """文本切分，确保切分后的块是合理的"""
    # 移除多余的空行，但保留正常的换行符 \n，防止文本变成一坨
    lines = text.split("\n")
    clean_text = "\n".join([line.strip() for line in lines if line.strip()])
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", ".", "!", "?", ";", ","]
    )
    chunks = text_splitter.split_text(clean_text)
    
    # 过滤掉长度小于 20 的无意义碎片
    filtered_chunks = [c for c in chunks if len(c.strip()) >= 20]
    return filtered_chunks