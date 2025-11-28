print("Hello")
# Import Library
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI  # ✅ Correct imports
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

# get api key
import os
os.environ["GOOGLE_API_KEY"] = "GEMINI-API-KEY"

# Step 1: Load documents and split them
loader = TextLoader("example.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

# Step 2: Embed documents using Gemini embeddings
embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = FAISS.from_documents(docs, embedding)

# Step 3: Initialize Gemini chat model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

# Step 4: Set up Retrieval QA chain
qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

# Step 5: Ask a question According to your document.
query = "Summarise the book?"
response = qa.run(query)
print(response)