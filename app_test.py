# pip install streamlit langchain lanchain-openai beautifulsoup4 python-dotenv chromadb
import shutil

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_groq import ChatGroq
# from langchain_groq import GroqEmbeddings
# from groq import GroqAPI
from langchain.vectorstores import Chroma
import os
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings




load_dotenv()
os.environ['GROQ_API_KEY'] = "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"
os.environ["OPENAI_API_KEY"] = "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"
print("Manually set key:", os.getenv("OPENAI_API_KEY"))
print("API Key Loaded:", os.getenv("GROQ_API_KEY"))
# groq_api = GroqAPI(api_key=os.getenv("GROQ_API_KEY"))
class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        # Return a list of vectors, not a NumPy array
        return self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True).tolist()

    def embed_query(self, text):
        return self.model.encode([text], convert_to_numpy=True)[0].tolist()
# Use Groq's model for embedding generation
def generate_embeddings(texts):
    embeddings = []
    for text in texts:
        response = groq_api.embed(text)  # You might need to adjust the API call
        embeddings.append(response['embedding'])  # Extract the embedding from the response
    return embeddings

# def get_vectorstore_from_url(url):
#     # get the text in document form
#     loader = WebBaseLoader(url)
#     document = loader.load()
    
#     # split the document into chunks
#     text_splitter = RecursiveCharacterTextSplitter()
#     document_chunks = text_splitter.split_documents(document)
    
#     # create a vectorstore from the chunks
#     # vector_store = Chroma.from_documents(document_chunks, OpenAIEmbeddings())
#     # vector_store = Chroma.from_documents(document_chunks, GroqEmbeddings(api_key=os.getenv("GROQ_API_KEY")))
#     # vector_store = Chroma.from_documents(document_chunks, generate_embeddings)
#     vector_store = Chroma.from_documents(document_chunks, SentenceTransformerEmbeddings())
#     vectorstore = Chroma.from_documents(
#     document_chunks,
#     SentenceTransformerEmbeddings(),
#     persist_directory="./chroma_db"  # You can change the directory path if needed
# )
#     return vector_store
# def get_vectorstore_from_url(url):
#     persist_dir = "./chroma_db"

#     if os.path.exists(persist_dir) and os.listdir(persist_dir):
#         # Load existing persisted vector store
#         vector_store = Chroma(
#             persist_directory=persist_dir,
#             embedding_function=SentenceTransformerEmbeddings()
#         )
#     else:
#         # Create a new vector store
#         loader = WebBaseLoader(url)
#         document = loader.load()
#         text_splitter = RecursiveCharacterTextSplitter()
#         document_chunks = text_splitter.split_documents(document)

#         vector_store = Chroma.from_documents(
#             document_chunks,
#             SentenceTransformerEmbeddings(),
#             persist_directory=persist_dir
#         )
#         vector_store.persist()  # 👈 Very important: explicitly persist it

#     return vector_store
# def get_vectorstore_from_url(url):
#     persist_dir = "./chroma_db"

#     # Clean up old DB if switching to a new website
#     if os.path.exists(persist_dir):
#         shutil.rmtree(persist_dir)

#     loader = WebBaseLoader(url)
#     document = loader.load()
#     text_splitter = RecursiveCharacterTextSplitter()
#     document_chunks = text_splitter.split_documents(document)

#     vector_store = Chroma.from_documents(
#         document_chunks,
#         SentenceTransformerEmbeddings(),
#         persist_directory=persist_dir
#     )
#     vector_store.persist()

#     return vector_store


# def get_vectorstore_from_url(website_url):
#     persist_directory = "/tmp/chroma_db"  # Writable location on Render
#     os.makedirs(persist_directory, exist_ok=True)

#     loader = WebBaseLoader(website_url)
#     document = loader.load()
#     text_splitter = RecursiveCharacterTextSplitter()
#     document_chunks = text_splitter.split_documents(document)

#     vector_store = Chroma.from_documents(
#         # documents,
#         # embedding=embedding_model,
#         document_chunks,
#         SentenceTransformerEmbeddings(),
#         persist_directory=persist_directory
#     )
#     return vector_store

from chromadb.config import Settings

def get_vectorstore_from_url(website_url):
    persist_directory = "/tmp/chroma_db"
    os.makedirs(persist_directory, exist_ok=True)

    loader = WebBaseLoader(website_url)
    document = loader.load()
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)

    chroma_settings = Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=persist_directory,
        anonymized_telemetry=False
    )

    vector_store = Chroma.from_documents(
        document_chunks,
        SentenceTransformerEmbeddings(),
        persist_directory=persist_directory,
        client_settings=chroma_settings  # ✅ Ensures everything goes to /tmp
    )
    vector_store.persist()
    return vector_store

def get_context_retriever_chain(vector_store):
    # llm = ChatOpenAI()
    llm = ChatGroq(
    temperature=0,
    model_name= "llama-3.3-70b-versatile", #"mixtral-8x7b-32768",  # or "llama3-8b-8192"
    groq_api_key="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" # os.getenv("GROQ_API_KEY")
)
    retriever = vector_store.as_retriever()
    
    prompt = ChatPromptTemplate.from_messages([
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
      ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    
    return retriever_chain
    
def get_conversational_rag_chain(retriever_chain): 
    
    # llm = ChatOpenAI()
    llm = ChatGroq(
    temperature=0,
    model_name= "llama-3.3-70b-versatile", #"mixtral-8x7b-32768",  # or "llama3-8b-8192"
    groq_api_key="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" # os.getenv("GROQ_API_KEY")
)
    prompt = ChatPromptTemplate.from_messages([
      ("system", "Answer the user's questions based on the below context:\n\n{context}"),
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
    ])
    
    stuff_documents_chain = create_stuff_documents_chain(llm,prompt)
    
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def get_response(user_input):
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    
    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })
    
    return response['answer']

# app config
# st.set_page_config(page_title="Chat with websites", page_icon="🤖")
# st.title("Chat with websites")

# # sidebar
# with st.sidebar:
#     st.header("Settings")
#     website_url = st.text_input("Website URL")

# if website_url is None or website_url == "":
#     st.info("Please enter a website URL")

# else:
#     # session state
#     if "chat_history" not in st.session_state:
#         st.session_state.chat_history = [
#             AIMessage(content="Hello, I am a bot. How can I help you?"),
#         ]
#     if "vector_store" not in st.session_state:
#         st.session_state.vector_store = get_vectorstore_from_url(website_url)    

#     # user input
#     user_query = st.chat_input("Type your message here...")
#     if user_query is not None and user_query != "":
#         response = get_response(user_query)
#         st.session_state.chat_history.append(HumanMessage(content=user_query))
#         st.session_state.chat_history.append(AIMessage(content=response))
        
       

#     # conversation
#     for message in st.session_state.chat_history:
#         if isinstance(message, AIMessage):
#             with st.chat_message("AI"):
#                 st.write(message.content)
#         elif isinstance(message, HumanMessage):
#             with st.chat_message("Human"):
#                 st.write(message.content)
# pip install streamlit langchain langchain-openai beautifulsoup4 python-dotenv chromadb sentence-transformers


# =================== STREAMLIT UI ===================

# Set page config
st.set_page_config(page_title="Chat with websites", page_icon="🤖")
st.title("Chat with websites")

# Sidebar for user to enter URL
with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("Website URL")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am a bot. How can I help you?")
    ]
if "last_website_url" not in st.session_state:
    st.session_state.last_website_url = None

# If a URL is provided
if website_url:
    if website_url != st.session_state.last_website_url:
        # Remove old Chroma DB to prevent reuse
        persist_directory = "/tmp/chroma_db"
        if os.path.exists(persist_directory):
            shutil.rmtree(persist_directory)

        # Rebuild vectorstore and update session state
        st.session_state.vector_store = get_vectorstore_from_url(website_url)
        st.session_state.last_website_url = website_url

        # Reset chat history
        st.session_state.chat_history = [
            AIMessage(content="Hello, I am a bot. How can I help you?")
        ]

    # Chat input
    user_query = st.chat_input("Type your message here...")
    if user_query:
        response = get_response(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))

    # Display chat history
    for message in st.session_state.chat_history:
        role = "AI" if isinstance(message, AIMessage) else "Human"
        with st.chat_message(role):
            st.write(message.content)
else:
    st.info("Please enter a website URL in the sidebar to begin.")
