# GenAI LLM Testing Project (Ollama + DeepEval + LangChain )

The project demonstrates GenAI testing - evaluating Large Language Model outputs using automated metrics.

It uses a local model served by Ollama with LangChain as the interface and DeepEval to run automated checks.

This is a small project you can run on your computer. It allows you to:
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

### Getting Started

**Terminal 1 — start the Ollama server (keep this running):**
```bash
ollama serve
```

**Terminal 2 — pull the model (only needed once):**
```bash
ollama pull deepseek-r1:8b
```

---

### Installation

> It’s best to run this in a fresh virtual environment.

```bash
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip

pip install -r requirements.txt
```

**Set your OpenAI API key** (used by DeepEval as the judge model):
```bash
export OPENAI_API_KEY=your-key-here
```

> Or create a `.env` file in the project root with `OPENAI_API_KEY=your-key-here` and run `source .env`

---

### Running

**Run the full evaluation (Terminal 2):**
```bash
python -m src.eval.run_eval
```

**Run pytest tests:**
```bash
pytest src/tests/test_langchain_eval.py -v
pytest src/tests/test_metrics_eval.py::test_answer_relevancy -v
pytest src/tests/test_metrics_eval.py::test_contextual_precision -v
```

Results are saved to `reports.json`.

---

### Note on API key