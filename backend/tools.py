from .rag import get_retriever,get_youtube_retriever,reciprocal_rank_fusion,rerank,THREAD_METADATA,THREAD_YOUTUBE_METADATA
from .config import embeddings,llm

import numpy as np
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from typing import TypedDict, Annotated,Dict,Optional,Any,Literal,List
import requests





duckduckgo=DuckDuckGoSearchRun(region="us-en")
@tool
def duckduckgo_search(query: str) -> str:
    """
    Search the web using DuckDuckGo.
    Use only for recent/current information.
    """
    results= duckduckgo.run(query)
    return f"""
    Web search results:
    {results}

    Use ONLY this information to answer.
    """

@tool
def calculator(first_num:float,second_num:float,operation:str)->dict:
    """
    Perform a basic arithmetic operation on two numbers.
    and make sure to convert both the num to float
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}
        
        return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}
    except Exception as e:
        return {"error": str(e)}
    

@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=C9PE94QUEW9VWGFM"
    r = requests.get(url)
    return r.json()

@tool
def rag_tool(query:str,thread_id:Optional[str]=None)->dict:

    """
    Retrieve relevant information from the uploaded PDF for this chat thread.
    Always include the thread_id when calling this tool.
    """

    retriever=get_retriever(thread_id)

    if not retriever:
        return{
            "error":"NO document indexed for this chat.Upload a PDF first",
            "query":query,
            }
    
    pdf_index = retriever["index"]
    pdf_chunks = retriever["chunks"]
    pdf_metadata = retriever["metadata"]
    pdf_bm25=retriever["bm25"]

    query_embedding=embeddings.embed_query(query)
    query_vector=np.array(
        [query_embedding],
        dtype=np.float32
    )

    tokenized_query=query.lower().split()
    bm25_results = pdf_bm25.get_top_n(tokenized_query, [chunk.page_content for chunk in pdf_chunks], n=20)

    distances, indices = pdf_index.search(query_vector, k=20)

    faiss_results = []
    
    

    for dist, idx in zip(distances[0], indices[0]):
        if dist < 1.5:      # tune threshold
            faiss_results.append(pdf_chunks[idx].page_content)

    # combined_results=faiss_results

    # for i in bm25_results:
    #     if i not in combined_results:
    #         combined_results.append(i)


    rrf_results = reciprocal_rank_fusion([faiss_results, bm25_results], k=60)[:5]
    
    final_results=rerank(query,rrf_results,5)

    return {
        "query":query,
        "context":"\n\n".join(final_results),
        "retrieved_chunks": final_results,
        "metadata":pdf_metadata,
        "source_file":THREAD_METADATA.get(str(thread_id),{}).get('filename')
    }


@tool
def youtube_rag_tool(query:str,thread_id:str)->dict:
    """
    Retrieve relevant information from the provided url,and answer the question of the user accordingly
    """
    retrievers=get_youtube_retriever(thread_id)
    if retrievers is None:
        return{
            "error":"NO youtube video indexed for this chat.Upload a PDF first",
            "query":query,
            }

    yt_chunks=retrievers['chunks']
    tokenized_query=query.lower().split()

    faiss_result=retrievers["faiss_retriever"].invoke(query)
    faiss_result=[res.page_content for res in faiss_result]
    bm25_result=retrievers["bm25"].get_top_n(tokenized_query, yt_chunks, n=20)
    bm25_result = [doc.page_content for doc in bm25_result]

    rrf_results = reciprocal_rank_fusion([faiss_result, bm25_result], k=60)[:5]
    merged_texts=rerank(query,rrf_results,top_k=5)

    # Rebuild metadata by matching text back to original chunks
    text_to_metadata = {chunk.page_content: chunk.metadata for chunk in yt_chunks}
    metadata = [text_to_metadata.get(t, {}) for t in merged_texts]

    return {
        "query": query,
        "context": "\n\n".join(merged_texts),
        "metadata": metadata,
        "source": THREAD_YOUTUBE_METADATA.get(str(thread_id), {}).get("url"),
    }





#---------------------------------------------------
# Make tool list
tools = [get_stock_price, duckduckgo_search, calculator,rag_tool,youtube_rag_tool]

# Make the LLM tool-aware
llm_with_tools = llm.bind_tools(tools)

