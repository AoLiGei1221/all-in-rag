from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import load_index_from_storage

# 1. 配置全局嵌入模型
Settings.embed_model = HuggingFaceEmbedding("BAAI/bge-small-zh-v1.5")

# 2. 创建示例文档
texts = [
    "张三是法外狂徒",
    "LlamaIndex是一个用于构建和查询私有或领域特定数据的框架。",
    "它提供了数据连接、索引和查询接口等工具。"
]
docs = [Document(text=t) for t in texts]

# 3. 创建索引并持久化到本地
index = VectorStoreIndex.from_documents(docs)
persist_path = "./llamaindex_index_store"


# persist data to disk, under the specified persist_path
index.storage_context.persist(persist_dir=persist_path)
print(f"LlamaIndex 索引已保存至: {persist_path}")

print("正在从本地加载索引...")
storage_context = index.storage_context.from_defaults(persist_dir=persist_path)
loaded_index = load_index_from_storage(storage_context)
print("索引加载成功！\n")

query = "FAISS是做什么的？"
retriever = loaded_index.as_retriever(similarity_top_k=1)
results = retriever.retrieve(query)

print(f"纯检索查询: '{query}'")
print("相似度最高的文档:")
for node in results:
    # LlamaIndex 的节点内容通过 node.text 或 node.get_content() 获取
    print(f"- {node.text} (相似度得分: {node.score:.4f})\n")