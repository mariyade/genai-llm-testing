import time
from deepeval.test_case import LLMTestCase
from deepeval.dataset import EvaluationDataset, Golden
from src.clients.llm_ollama import ask
from src.eval.metrics import make_metrics

def test_answer_relevancy():
    metrics = make_metrics()
    metric = metrics["answer_relevancy"]

    test_case = LLMTestCase(
        input="Who is the current president of the United States of America?",
        actual_output="Joe Biden",
        retrieval_context=["Joe Biden serves as the current president of America."],
        expected_output="Joe Biden"
    )

    start = time.time()
    metric.measure(test_case)
    end = time.time()

    print(f"Answer Relevancy completed in {end - start:.2f} seconds")
    print("Score:", metric.score)
    assert metric.score is not None

def test_contextual_precision():
    metrics = make_metrics()
    metric = metrics["context_precision"]

    test_case = LLMTestCase(
        input="Who is the current president of USA in 2024?",
        actual_output="Donald Trump",
        retrieval_context=["Donald Trump serves as the current president of America."],
        expected_output="Donald Trump is the current president of America."
    )

    metric.measure(test_case)
    print("Score:", metric.score)
    assert metric.score is not None

def test_batch_dataset_eval():
    metrics = make_metrics()
    answer_rel = metrics["answer_relevancy"]
    context_prec = metrics["context_precision"]

    goldens_for_dataset = [
        Golden(input="Who is the current president of the United States of America?", expected_output="Joe Biden"),
        Golden(input="Who is the current president of USA in 2024?", expected_output="Donald Trump")
    ]

    dataset = EvaluationDataset(goldens=goldens_for_dataset)
    results = dataset.evaluate(metrics=[answer_rel, context_prec], return_results=True)

    for metric_obj, metric_result in results.items():
        metric_name = getattr(metric_obj, "name", metric_obj.__class__.__name__)
        print(f"\nMetric: {metric_name}")
        print("Score:", metric_result.score)
