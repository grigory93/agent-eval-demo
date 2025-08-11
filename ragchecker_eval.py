from ragchecker import RAGResults, RAGChecker
from ragchecker.metrics import all_metrics
import os
import json

def run_ragchecker_eval(agent_results_file):

    # INSERT_YOUR_CODE
    if not os.path.exists(agent_results_file):
        raise FileNotFoundError(f"File not found: {agent_results_file}")

    # initialize ragresults from json/dict
    with open(agent_results_file) as fp:
        rag_results = RAGResults.from_json(fp.read())

    # set-up the evaluator
    # extractor_model_name = "ai21/jamba-mini-1.7"
    # checker_model_name = "ai21/jamba-mini-1.7"
    extractor_model_name = "openai/gpt-4o-mini"
    checker_model_name = "openai/gpt-4o-mini"

    evaluator = RAGChecker(
        extractor_name=extractor_model_name,
        checker_name=checker_model_name,
        batch_size_extractor=10,
        batch_size_checker=10,
    )

    # evaluate results with selected metrics or certain groups, e.g., retriever_metrics, generator_metrics, all_metrics
    evaluator.evaluate(rag_results, ["overall_metrics", "retriever_metrics", "generator_metrics"])
    # evaluator.evaluate(rag_results, ["overall_metrics"])

    # Save evaluated RAGResults to a new JSON file with '_eval' suffix
    base, ext = os.path.splitext(agent_results_file)
    eval_file = f"{base}_eval{ext}"

    with open(eval_file, "w") as f:
        f.write(rag_results.to_json())

    return rag_results, eval_file

"""Output
RAGResults(
  2 RAG results,
  Metrics:
  {
    "overall_metrics": {
      "precision": 76.4,
      "recall": 62.5,
      "f1": 68.3
    },
    "retriever_metrics": {
      "claim_recall": 61.4,
      "context_precision": 87.5
    },
    "generator_metrics": {
      "context_utilization": 87.5,
      "noise_sensitivity_in_relevant": 19.1,
      "noise_sensitivity_in_irrelevant": 0.0,
      "hallucination": 4.5,
      "self_knowledge": 27.3,
      "faithfulness": 68.2
    }
  }
)
"""