import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_groq import ChatGroq
from backend.tools import rag_tool
from backend.config import embeddings
from evaluation.eval_dataset import eval_rag_data
llm = ChatGroq(model="openai/gpt-oss-120b")

ragas_llm = LangchainLLMWrapper(llm)
ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)

for metric in [faithfulness, answer_relevancy, context_recall]:
    metric.llm = ragas_llm
    metric.embeddings = ragas_embeddings

# ── Build eval samples ──────────────────────────────────────────────────────
THREAD_ID = "22e779de-ad75-4fca-a024-345f50d4eb5d"

eval_samples = []

for i, sample in enumerate(eval_rag_data):

    rag_tool_response = rag_tool.func(
        query=sample["question"],
        thread_id=THREAD_ID
    )

    if "error" in rag_tool_response:
        print(f"[{i}] Skipped — {rag_tool_response['error']}")
        continue

    contexts = rag_tool_response["retrieved_chunks"]

    prompt = f"""You are a helpful assistant.
Answer ONLY using the provided context.

Context:
{rag_tool_response['context']}

Question:
{sample['question']}
"""

    try:
        answer = llm.invoke(prompt).content
        eval_samples.append({
            "question":     sample["question"],
            "answer":       answer,
            "ground_truth": sample["ground_truth"],
            "contexts":     contexts,
        })
        print(f"[{i}] collected")
    except Exception as e:
        print(f"[{i}] Failed to get answer: {e}")

# ── Run ragas evaluation ────────────────────────────────────────────────────
if not eval_samples:
    print("No samples collected — check THREAD_ID or eval_dataset.")
else:
    dataset = Dataset.from_list(eval_samples)

    result = evaluate(
        dataset,
        metrics=[faithfulness, context_recall, answer_relevancy],
    )

    print("\n=== RAGAS Evaluation Results ===")
    print(result)



# {'faithfulness': 0.9215, 'context_recall': 0.9467, 'answer_relevancy': 0.8272}


# result for rrf +crossencoder reranking
# {'faithfulness': 0.9641, 'context_recall': 0.8833, 'answer_relevancy': 0.8233}


# result for hybrid search + rrf
