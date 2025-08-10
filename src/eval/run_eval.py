import json
from deepeval.test_case import LLMTestCase
from deepeval.dataset import EvaluationDataset
from src.clients.llm_ollama import ask
from src.eval.metrics import make_metrics

def main():
    tc1 = LLMTestCase(
        input="Say 'hello' in one short sentence.",
        actual_output=ask("Say 'hello' in one short sentence."),
        expected_output="Hello."
    )
    tc2 = LLMTestCase(
        input="List headings of common LLM biases only.",
        actual_output=ask("List headings of common LLM biases only."),
        retrieval_context=["Gender Bias, Racial Bias, Ethnic Bias, Religious Bias, Political Bias, Cultural Bias, Educational Bias, Linguistic Bias, Ageism, Economic Bias, Nationalist Bias"],
        expected_output="Gender Bias; Racial Bias; Ethnic Bias; Religious Bias; Political Bias; Cultural Bias; Educational Bias; Linguistic Bias; Ageism; Economic Bias; Nationalist Bias"
    )

    dataset = EvaluationDataset(test_cases=[tc1, tc2])
    metrics = list(make_metrics().values())[:2] 
    results = dataset.evaluate(metrics=metrics, return_results=True)

    report = {
        getattr(k, "name", k.__class__.__name__): {
            "score": v.score,
            "success": getattr(v, "success", None),
            "details": getattr(v, "score_breakdown", None)
        }
        for k, v in results.items()
    }

    with open("reports.json", "w") as f:
        json.dump(report, f, indent=2)

    print("Evaluation complete. Results written to reports.json")

if __name__ == "__main__":
    main()
