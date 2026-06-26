from .config import cross_encoder,base_path,embeddings,base_path_yt

import os
from typing import TypedDict, Annotated,Dict,Optional,Any,Literal,List
from langchain_core.messages import BaseMessage,HumanMessage,SystemMessage,AIMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import YoutubeLoader

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
import faiss
import numpy as np
import pickle
import json
import tempfile
from datetime import datetime, timezone

import bm25s
from rank_bm25 import BM25Okapi


#Cache
THREAD_RETRIEVERS:Dict[str,Any]={}
THREAD_METADATA:Dict[str,dict]={}

THREAD_YOUTUBE_VECTORSTORE:Dict[str,Any]={}
THREAD_YOUTUBE_METADATA:Dict[str,dict]={}



def reciprocal_rank_fusion(result_lists: list[list[str]], k: int = 60) -> list[str]:
    """
    Merge multiple ranked lists of text chunks using Reciprocal Rank Fusion.
    result_lists: list of lists, each list is ranked from most to least relevant.
    Returns a single merged list ranked by RRF score.
    """
    scores: dict[str, float] = {}

    for ranked_list in result_lists:
        for rank, doc_text in enumerate(ranked_list):
            if doc_text not in scores:
                scores[doc_text] = 0.0
            scores[doc_text] += 1.0 / (k + rank + 1)  # rank+1 so rank 0 → position 1

    # Sort by descending RRF score
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_text for doc_text, _ in ranked]


def rerank(query,docs,top_k=5):

    pairs=[(query,doc) for doc in docs]

    scores=cross_encoder.predict(pairs)

    ranked=sorted(
        zip(docs,scores),
        key=lambda x: x[1],
        reverse=True
    )
    return [doc for doc,score in ranked[:top_k]]



def get_retriever(thread_id:Optional[str]):
    """ Fetch the retriever for a thread if its available"""
    if thread_id and thread_id in THREAD_RETRIEVERS:
        return THREAD_RETRIEVERS[thread_id]
    
    file_path = os.path.join(base_path, thread_id)
    if not os.path.exists(file_path):
        return None

    #Disk load
    pdf_index=faiss.read_index(
        os.path.join(file_path,"index.faiss")
    )

    with open(os.path.join(file_path,"index.pkl"),"rb") as f:
        pdf_chunks=pickle.load(f)
    with open(os.path.join(file_path,"index.json"),"r") as f:
        metadata=json.load(f)
    with open(os.path.join(file_path,"bm25.pkl"),"rb") as f:
        bm25=pickle.load(f)

    data = {
        "index": pdf_index,
        "chunks": pdf_chunks,
        "metadata": metadata,
        "bm25":bm25,
    }

    # Store in cache
    THREAD_RETRIEVERS[thread_id] = data

    return data

def ingest_pdf(file_bytes:bytes,thread_id:str,filename:Optional[str]=None)->dict:
    """
    Build a FAISS retriever for the uploaded PDF and store it for the thread.

    Returns a summary dict that can be surfaced in the UI.
    """
    if not file_bytes:
        raise ValueError("No bytes received for Ingestion")
    
    with tempfile.NamedTemporaryFile(delete=False,suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path=temp_file.name

    try:
        loader=PyPDFLoader(temp_path)
        docs=loader.load()

        splitter=RecursiveCharacterTextSplitter(
            chunk_size=1000,chunk_overlap=200,separators=["\n\n", "\n", " ", ""]
        )
        chunks=splitter.split_documents(docs)
        if len(docs)==0 or len(chunks)==0:
            raise ValueError("No text could be extracted from the PDF.")
        
        texts=[doc.page_content for doc in chunks]

        #bm25s
        tokenized_texts=[sent.lower().split() for sent in texts]
        bm25=BM25Okapi(tokenized_texts)
        
        
        document_embeddings=embeddings.embed_documents(texts)
        
        pdf_index = faiss.IndexFlatL2(len(document_embeddings[0]))
        pdf_index_embeddings= pdf_index.add(np.array(document_embeddings,dtype=np.float32))

        file_path=fr"C:\Users\sudha\Desktop\My_workspace\langgraph\langgraph_chatbot\Multi-Source_AI_Assistant\storage\pdf\{thread_id}"

        os.makedirs(file_path, exist_ok=True)
        
        faiss.write_index(pdf_index, os.path.join(file_path, "index.faiss"))
        
        with open(os.path.join(file_path, "bm25.pkl"),"wb") as f:
            pickle.dump(bm25,f)
        
        with open( os.path.join(file_path, "index.pkl"),"wb") as f:
            pickle.dump(chunks,f)


        metadata={
            "thread_id":thread_id,
            "filename":filename,
            "documents": len(docs),
            "num_chunks": len(chunks),
            "created_at": datetime.now(timezone.utc).isoformat()
        }

        THREAD_METADATA[str(thread_id)] = metadata
        with open(os.path.join(file_path,"index.json"),"w") as f:
            json.dump(metadata, f, indent=2)

        THREAD_RETRIEVERS[thread_id] = {
            "index": pdf_index,
            "chunks": chunks,
            
            "metadata": metadata,
            "bm25":bm25,
        }
        

        return{
            "filename":filename or os.path.basename(temp_path),
            "documents":len(docs),
            "chunks":len(chunks),

        }
    
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass

def thread_has_document(thread_id: str) -> bool:
    BASE_PATH=fr"C:\Users\sudha\Desktop\My_workspace\langgraph\langgraph_chatbot\Multi-Source_AI_Assistant\storage\pdf"
    path = os.path.join(BASE_PATH, thread_id, "index.faiss")
    return os.path.exists(path)
def thread_document_metadata(thread_id: str) -> dict:
    return THREAD_METADATA.get(str(thread_id), {})

# Youtube

def get_youtube_retriever(thread_id:Optional[str]):
    """ Fetch the retriever for a thread if its available"""
    if thread_id and thread_id in THREAD_YOUTUBE_VECTORSTORE:
        retriever=THREAD_YOUTUBE_VECTORSTORE[thread_id]["faiss_vector_store"].as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )
        chunks=THREAD_YOUTUBE_VECTORSTORE[thread_id]["chunks"]
        bm25=THREAD_YOUTUBE_VECTORSTORE[thread_id]["bm25"]
        return {"faiss_retriever":retriever,"bm25":bm25,"chunks":chunks}

    file_path=os.path.join(base_path_yt,thread_id)
    if not file_path:
        return None
    
    vector_store = FAISS.load_local(
    file_path,
    embeddings,
    allow_dangerous_deserialization=True
    )

    retriever=vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k":20,},
        
    )

    with open(os.path.join(file_path,"bm25.pkl"),"rb") as f:
        bm25=pickle.load(f)

    with open(os.path.join(file_path,"chunks.pkl"),"rb") as f:
        chunks=pickle.load(f)

    THREAD_YOUTUBE_VECTORSTORE[str(thread_id)] = {"faiss_vector_store":vector_store,"bm25":bm25,"chunks":chunks}
    return {"faiss_retriever":retriever,"bm25":bm25,"chunks":chunks}

def ingest_youtube(url:str,thread_id:str)->dict:
    loader=YoutubeLoader.from_youtube_url(
    url,
    add_video_info=False,
    language=["en","hi","kn","ta"]
    )

    docs=loader.load()
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    chunks=text_splitter.split_documents(docs)
    file_path=os.path.join(base_path_yt,thread_id)

    vector_store=FAISS.from_documents(chunks,embeddings)
    vector_store.save_local(file_path)

    tokenized_texts=[chunk.page_content.lower().split() for chunk in chunks]
    bm25=BM25Okapi(tokenized_texts)
    with open(os.path.join(file_path,"bm25.pkl"),"wb") as f:
        pickle.dump(bm25,f)
    with open(os.path.join(file_path,"chunks.pkl"),"wb") as f:
        pickle.dump(chunks,f)

    # retriever=vector_store.as_retriever(search_type="similarity",search_kwargs={"k":4})
    THREAD_YOUTUBE_VECTORSTORE[str(thread_id)]={"faiss_vector_store":vector_store,"bm25":bm25,"chunks":chunks}
    THREAD_YOUTUBE_METADATA[str(thread_id)]={
        'url':url,
        "chunks":len(chunks),
        
    }

    return {
        'url':url,
        'chunks':len(chunks)
    }


def thread_has_youtube(thread_id: str) -> bool:
    return str(thread_id) in THREAD_YOUTUBE_VECTORSTORE

