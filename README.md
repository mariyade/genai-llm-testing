# GenAI LLM Testing Project (Ollama + DeepEval + LangChain )

The project demonstrates GenAI testing - evaluating Large Language Model outputs using automated metrics.

It uses a local model served by Ollama with LangChain as the interface and DeepEval to run automated checks.

This is a small project allows you can run locally: 
- connect to a language model using LangChain and Ollama
- send some sample questions or prompts to the model
- automatically score the model’s answers using built-in metrics
- check answers against a “Golden” dataset to see if they match

---

### Metrics Evaluation using DeepEval:

- AnswerRelevancyMetric
- ContextualPrecisionMetric
- BiasMetric
- GEval for custom bias check


PyTest Test Cases for:

- Smoke-testing the LLM connection
- Individual metric runs
- Batch evaluation with golden datasets

Evaluation Script (src/eval/run_eval.py) to run all cases and export results

---
### Installation

> It’s best to run this in a fresh virtual environment.

```bash
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip

pip install -r requirements.txt
