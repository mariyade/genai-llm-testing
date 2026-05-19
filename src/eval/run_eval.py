import json
from deepeval.test_case import LLMTestCase
from deepeval.dataset import EvaluationDataset
from src.clients.llm_ollama import ask
from src.eval.metrics import make_metrics

def main():
    tc1 = LLMTestCase(
        input="Say 'hello' in one short sentence.",
        actual_output=ask("Say 'hello' in one short sentence."),
        expected_output="Hello.",
        retrieval_context=["A greeting is a polite word or phrase used to acknowledge someone."]
    )
    tc2 = LLMTestCase(
        input="List headings of common LLM biases only.",
        actual_output=ask("List headings of common LLM biases only."),
        retrieval_context=["Gender Bias, Racial Bias, Ethnic Bias, Religious Bias, Political Bias, Cultural Bias, Educational Bias, Linguistic Bias, Ageism, Economic Bias, Nationalist Bias"],
        expected_output="Gender Bias; Racial Bias; Ethnic Bias; Religious Bias; Political Bias; Cultural Bias; Educational Bias; Linguistic Bias; Ageism; Economic Bias; Nationalist Bias"
    )

    dataset = EvaluationDataset(test_cases=[tc1, tc2])
    metrics = list(make_metrics().values())[:2] 
    dataset.evaluate(metrics=metrics)

    report = {
        getattr(metric, "name", metric.__class__.__name__): {
            "score": metric.score,
            "success": getattr(metric, "success", None),
            "details": getattr(metric, "score_breakdown", None)
        }
        for metric in metrics
    }

    with open("reports.json", "w") as f:
        json.dump(report, f, indent=2)

    print("Evaluation complete. Results written to reports.json")

if __name__ == "__main__":
    main()
