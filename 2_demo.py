# Import necessary library
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY", "Your-api-key"))

# Sample knowledge base
documents = [
    "My name is Chaman GamanBhai Gajera.",
    "I am Senior officer of Security Department.",
    "Artificial Intelligence (AI) is the simulation of human intelligence in machines.",
    "Retrieval-Augmented Generation (RAG) enhances LLMs by combining retrieval with generation."
]

# Initialize embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Embed documents
document_embeddings = embedder.encode(documents)

def retrieve_documents(query, top_k=2): # top-k for retrieve most related document
    """Retrieve top-k relevant documents for the query."""
    query_embedding = embedder.encode([query])[0]
    similarities = cosine_similarity([query_embedding], document_embeddings)[0]
    top_k_indices = np.argsort(similarities)[-top_k:][::-1]
    return [documents[idx] for idx in top_k_indices]

def generate_response(query, retrieved_docs):
    """Generate a response using Gemini API with retrieved documents as context."""
    context = "\n".join(retrieved_docs)
    prompt = f"Query: {query}\nContext: {context}\nAnswer in a concise and clear manner:"
    
    try:
        # Initialize the Gemini model (e.g., gemini-1.5-flash (Which is free version of gemini))
        model = genai.GenerativeModel('gemini-1.5-flash') # define the model which want to use
        response = model.generate_content(prompt) # model will generate the content according prompt
        return response.text 
    except Exception as e: 
        return f"Error generating response: {str(e)}"

# Example usage
def demonstrate_rag():
    query = "What imy position?" # User Query to model
    print("Query:", query)
    
    # Step 1: Retrieve relevant documents
    retrieved_docs = retrieve_documents(query) 
    print("\nRetrieved Documents:")
    for i, doc in enumerate(retrieved_docs, 1):
        print(f"{i}. {doc}")
    
    # Step 2: Generate response
    response = generate_response(query, retrieved_docs)
    print("\nGenerated Response:", response)

# Run the demonstration
if __name__ == "__main__":
    demonstrate_rag()