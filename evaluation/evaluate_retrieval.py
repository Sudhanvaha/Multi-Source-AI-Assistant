from .eval_dataset import eval_rag_data
from backend.tools import rag_tool


def token_overlap(text: str,expected:str) ->float:

    expected_tokens=set(expected.lower().split())
    chunk_tokens=set(text.lower().split())

    if not expected_tokens:
        return 0.0
    return len(expected_tokens & chunk_tokens)/len(expected_tokens)

OVERLAP_THRESHOLD=.4

def recall_at_k(results):
    correct=0

    for item in results:

        expected=item["expected_answer"]
        found=False

        for chunk in item["retrieved_chunks"]:

            if token_overlap(chunk,expected)>=OVERLAP_THRESHOLD:
                found=True
                break

        if found:
            correct+=1

    return correct/len(results)


def calculate_mrr(results):
    
    scores=[]

    for item in results:

        expected=item["expected_answer"]
        rank=None

        for i,chunk in enumerate(item["retrieved_chunks"],start=1):

            if token_overlap(chunk,expected)>=OVERLAP_THRESHOLD:
                rank=i
                break

        if rank:
            scores.append(1/rank)
        else:
            scores.append(0)

    return sum(scores)/len(results)


eval_results=[]

for i,sample in enumerate(eval_rag_data):

    # response=rag_tool.invoke({"query":sample["question"],"thread_id":f"{i}"})
    response = rag_tool.func(
    query=sample["question"],
    thread_id=str("22e779de-ad75-4fca-a024-345f50d4eb5d")
    )

    if "error" in response:
        print(f"[{i}] Skipped — {response['error']}")
        continue

    # print(response)

    eval_results.append({
        "question": sample["question"],
        "expected_answer": sample["ground_truth"],
        "retrieved_chunks": response["retrieved_chunks"]
    }
    )


if not eval_results:
    print("No results to evaluate. Make sure a PDF is indexed for each thread_id.")
else:
    print("Recall@5:", recall_at_k(eval_results))
    print("MRR:     ", calculate_mrr(eval_results))


# for faiss+bm25+rrf+rerank(crossencoder),threshold=.4

    # Recall@5: 0.65
    # MRR:      0.43499999999999994

# for faiss+bm25+rrf+rerank(crossencoder),threshold=.3

    # Recall@5: 0.9
    # MRR:      0.8


# faiss results
# Recall@5: 0.71
# MRR:      0.6145

# # bm25 results
# Recall@5: 0.71
# MRR:      0.6145

# faiss+ bm25 results
# Recall@5: 0.71
# MRR:      0.6145

#rrf results
# Recall@5: 0.71
# MRR:      0.615

#final rrf+reranking 
# Recall@5: 0.71
# MRR:      0.615