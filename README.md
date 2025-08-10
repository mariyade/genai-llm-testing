# GenAI LLM Testing Project with Ollama, DeepEval and LangChain 

The project demonstrates GenAI testing - evaluating Large Language Model outputs using automated metrics.

It uses a local model served by Ollama with LangChain as the interface and DeepEval to run automated checks.

It's a designed as a minimum, job-ready example that can be run locally to:
- Connect to a LLM via LangChain and Ollama
- Run predefined prompts
- Score responses and automated metrics
- Use Golden datasets to validate answers against expected outputs

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
