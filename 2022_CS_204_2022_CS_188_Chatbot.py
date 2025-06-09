import os
import streamlit as st
import google.generativeai as genai
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

# Initialize Pinecone client
api_key = ""  # Replace with your Pinecone API key
environment = "us-east-1"  # Set to your Pinecone environment region
pc = Pinecone(api_key=api_key, environment=environment)

# Connect to the Pinecone index
index_name = "hiv"  # Replace with your Pinecone index name
index = pc.Index(index_name)

# Initialize the SentenceTransformer model (BERT-based)
model = SentenceTransformer('bert-base-uncased')  # You can use other models if needed

# Function to generate embeddings from the text
def generate_embeddings(text):
    return model.encode([text])[0]  # Generate embedding for the input text

# Function to query Pinecone for relevant information
def query_pinecone(query, top_k=3):
    query_embedding = generate_embeddings(query)
    query_results = index.query(
        vector=query_embedding.tolist(),  # Convert numpy array to list
        top_k=top_k,
        include_metadata=True
    )
    
    # Get the chunks of text without book names (no need to return book names here)
    results = [result['metadata']['text'] for result in query_results['matches']]
    return results

# Function to generate a response using the Gemini API
def generate_response(prompt, api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error generating response: {e}")
        return None

# Function to generate a response for user queries with relevant context
def generate_response_with_context(relevant_chunks, user_query, api_key):
    detailed_prompt = (
        f"You are a helpful, knowledgeable assistant with expertise in HIV and AIDS. "
        f"Additionally, based on the user query, here are some relevant insights: {relevant_chunks}. "
        f"Provide an empathetic and helpful response to the following user query: {user_query}. "
    )
    
    return generate_response(detailed_prompt, api_key)

# Streamlit interface
def main():
    st.title('HIV Information Chatbot')

    # Step 1: Display a brief description of HIV and AIDS (Static Information)
    st.write("HIV and AIDS are serious conditions, but they can be managed with the right treatment and care.")
    st.write("Here are some important topics related to HIV: prevention, treatment options, and managing health with HIV.")

    # Step 2: User input section for queries
    user_query = st.text_input("Ask a question about HIV or AIDS:")

    if user_query:
        # Step 3: Query Pinecone for relevant chunks
        relevant_chunks = query_pinecone(user_query)

        # Format relevant chunks to pass to Gemini, but without book names
        formatted_chunks = "\n".join(relevant_chunks)

        # Step 4: Generate the response using the Gemini API with context from Pinecone results
        api_key = ""  # Replace with your Gemini API key
        response = generate_response_with_context(formatted_chunks, user_query, api_key)

        # Step 5: Display the response from the chatbot
        if response:
            st.write(f"Chatbot Response: {response}")
        else:
            st.write("Failed to generate a response.")

if __name__ == "__main__":
    main()
