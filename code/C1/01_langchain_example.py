import os
# hugging face镜像设置，如果国内环境无法使用启用该设置
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# load environmental variables
load_dotenv()

# Define the path
markdown_path = "../../data/C1/markdown/easy-rl-chapter1.md"

# load local markdown file so we define loader and load the markdown file
loader = UnstructuredMarkdownLoader(markdown_path)
docs = loader.load()

# Chunking
# Long docs can be splitted into small and manageable chunks
# 递归字符分割策略，这里没有定义parameter=》所以是默认最大程度保留文本的语义结构
text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=200)
chunks = text_splitter.split_documents(docs)


# Build Index
# 中文 Embedding Model
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
  
# Build Vector
# 把分割好的chunks通过初始化好的embedding model转换成向量表达
# send chunks into embedding model => turn them into vectors 
#                                   => store them into vectorstore
vectorstore = InMemoryVectorStore(embeddings)
vectorstore.add_documents(chunks)

# 提示词模板
prompt = ChatPromptTemplate.from_template("""请根据下面提供的上下文信息来回答问题。
请确保你的回答完全基于这些上下文。
如果上下文中没有足够的信息来回答问题，请直接告知：“抱歉，我无法根据提供的上下文找到相关信息来回答此问题。”

上下文:
{context}

问题: {question}

回答:"""
                                          )

# 配置大语言模型

# 使用 AIHubmix
llm = ChatOpenAI(
    model="coding-glm-5.1-free",
    temperature=0.7,
    max_tokens=4096,
    api_key=os.getenv("AIHUBMIX_API_KEY"),
    base_url="https://aihubmix.com/v1"
)

# llm = ChatOpenAI(
#     model="deepseek-chat",
#     temperature=0.7,
#     max_tokens=4096,
#     api_key=os.getenv("DEEPSEEK_API_KEY"),
#     base_url="https://api.deepseek.com"
# )

# User query
question = "文中举了哪些例子？"

# In vectorstore, we search three chunks that have the highest similarity
retrieved_docs = vectorstore.similarity_search(question, k=3)

# prepare the context(上下文) and use "\n\n"分隔各个块形成最终的上下文信息供LLM参考
docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

# invoke LLM and push question and context inside
answer = llm.invoke(prompt.format(question=question, context=docs_content))
print(answer.content)
