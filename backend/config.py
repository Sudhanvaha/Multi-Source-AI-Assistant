from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder


load_dotenv()


base_path=fr"C:\Users\sudha\Desktop\My_workspace\langgraph\langgraph_chatbot\Multi-Source_AI_Assistant\storage\pdf"
base_path_yt=fr"C:\Users\sudha\Desktop\My_workspace\langgraph\langgraph_chatbot\Multi-Source_AI_Assistant\storage\youtube"

# LLMs
memory_llm=ChatGroq(model="llama-3.1-8b-instant",temperature=0)

llm = ChatGroq(model="llama-3.3-70b-versatile")


embeddings=HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    
)
cross_encoder = CrossEncoder(
    "BAAI/bge-reranker-base"
)



# DB URIs
DB_URI="postgresql://postgres:postgres@localhost:5432/postgres_chatbot"
DB_URI_LTM="postgresql://postgres:postgres@localhost:5442/postgres_ltm_store"