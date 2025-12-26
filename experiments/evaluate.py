from ragas.metrics.base import Metric
import json
import os
from datasets import Dataset
from google import genai
from ragas import evaluate
from ragas.llms import llm_factory
from ragas.embeddings import GoogleEmbeddings
from ragas.metrics import (
    AnswerCorrectness,
    ContextPrecision,
    ContextRecall,
    Faithfulness
)
from dotenv import load_dotenv

load_dotenv()


client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
llm = llm_factory("gemini-3-flash-preview", provider="google", client=client)
embeddings = GoogleEmbeddings(client=client, model="gemini-embedding-001")


def main():
    with open("experiment_results.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    dataset = Dataset.from_list(data)
    metrics = [
        ContextPrecision(llm=llm),
        ContextRecall(llm=llm),
        Faithfulness(llm=llm),
        AnswerCorrectness(llm=llm, embeddings=embeddings)
    ]

    results = evaluate(dataset, metrics=metrics)
    return results


if __name__ == "__main__":
    results = main()
    results = results.to_pandas()
    results.to_csv("evaluation_results.csv")
