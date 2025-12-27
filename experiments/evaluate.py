import asyncio
import json
import os
import time
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory
from ragas.cache import DiskCacheBackend
from ragas.metrics.collections import (
    AnswerCorrectness,
    ContextPrecision,
    ContextRecall,
    Faithfulness
)
from dotenv import load_dotenv

load_dotenv()


client = AsyncOpenAI(
    api_key=os.environ.get("GOOGLE_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

embeddings_cache = DiskCacheBackend()

llm = llm_factory(
    "gemini-2.5-flash",
    provider="openai",
    client=client,
    max_tokens=1000000,
)

embeddings = embedding_factory(
    "google", model="gemini-embedding-001", client=client, cache=embeddings_cache)


async def get_context_precision_scorer(user_input, reference, retrieved_contexts):
    scorer = ContextPrecision(llm=llm)
    result = await scorer.ascore(
        user_input=user_input,
        reference=reference,
        retrieved_contexts=retrieved_contexts,
    )
    return result.value


async def get_context_recall_scorer(user_input, reference, retrieved_contexts):
    scorer = ContextRecall(llm=llm)
    result = await scorer.ascore(
        user_input=user_input,
        retrieved_contexts=retrieved_contexts,
        reference=reference,
    )
    return result.value


async def get_faithfulness_scorer(user_input, response, retrieved_contexts):
    scorer = Faithfulness(llm=llm)
    result = await scorer.ascore(
        user_input=user_input,
        response=response,
        retrieved_contexts=retrieved_contexts,
    )
    return result.value


async def get_answer_correctness_scorer(user_input, response, reference):
    scorer = AnswerCorrectness(llm=llm, embeddings=embeddings)
    result = await scorer.ascore(
        user_input=user_input,
        response=response,
        reference=reference,
    )
    return result.value


async def ragas_evaluate(data):
    [context_precision, context_recall, faithfulness, answer_correctness] = await asyncio.gather(
        get_context_precision_scorer(
            data["question"], data["reference"], data["contexts"]
        ),
        get_context_recall_scorer(
            data["question"], data["reference"], data["contexts"]
        ),
        get_faithfulness_scorer(
            data["question"], data["answer"], data["contexts"]
        ),
        get_answer_correctness_scorer(
            data["question"], data["answer"], data["reference"]
        ),
    )

    return {
        **data,
        "context_precision": context_precision,
        "context_recall": context_recall,
        "faithfulness": faithfulness,
        "answer_correctness": answer_correctness
    }


async def main():
    with open("experiment_results.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)

    start_idx = 0
    results = []
    if os.path.exists("evaluation_results.json"):
        with open("evaluation_results.json", "r", encoding="utf-8") as f:
            results = json.load(f)
            start_idx = len(results)
            print(f"Resuming from index {start_idx}")

    dataset_len = len(dataset)
    batch_size = 10
    print(
        f"Evaluating {dataset_len - start_idx} remaining examples in batches of {batch_size}...")

    for i in range(start_idx, dataset_len, batch_size):
        batch_start = time.time()
        batch = dataset[i:i+batch_size]
        batch_results = await asyncio.gather(*[
            ragas_evaluate(row, idx=i+j) for j, row in enumerate(batch)
        ])
        results.extend(batch_results)
        print(
            f"=== Batch done: {time.time() - batch_start:.1f}s | {len(results)}/{dataset_len} ===")

        if len(results) % 100 == 0:
            with open("evaluation_results.json", "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
            print(f"Checkpoint saved at {len(results)}")

    return results


if __name__ == "__main__":
    results = asyncio.run(main())
    with open("evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print("Evaluation results saved to evaluation_results.json")
