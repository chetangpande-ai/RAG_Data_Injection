import os
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from dotenv import load_dotenv


# Load variables from .env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("❌ OPENAI_API_KEY not found. Please set it in your .env file")

print("✅ Key available:", True)

# 1. Load all PDFs from directory
loader = DirectoryLoader(
    "candidate_profiles/",
    glob="*.pdf",
    loader_cls=PyPDFLoader
)

documents = loader.load()


# 2. Split docs into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)


# 3. Create embeddings + FAISS store
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(chunks, embeddings)

# 4. Create retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})



# 5. Setup RAG pipeline
llm = ChatOpenAI(model="gpt-3.5-turbo",api_key=OPENAI_API_KEY)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# 6. Ask a question across all reports
query = "What is my cholesterol trend across all years?"
result = qa_chain.invoke(query)

print("🔹 Answer:", result["result"])
print("\n📂 Sources:")
for doc in result["source_documents"]:
    print("-", doc.metadata.get("source"), "->", doc.page_content[:150], "...")
