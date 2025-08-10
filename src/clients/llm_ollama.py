from langchain_ollama import ChatOllama

def get_llm(model="deepseek-r1:8b", temperature=0.5, num_predict=250, reasoning=True):
    return ChatOllama(
        base_url="http://localhost:11434",
        model=model,
        temperature=temperature,
        num_predict=num_predict,
        reasoning=reasoning,
    )

def ask(prompt: str, **kwargs) -> str:
    llm = get_llm(**kwargs)
    resp = llm.invoke(prompt)
    return resp.content.strip()
