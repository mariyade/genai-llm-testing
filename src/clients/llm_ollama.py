from langchain_ollama import ChatOllama

def get_llm(model="deepseek-r1:8b", temperature=0.5, num_predict=250, reasoning=True):
    return ChatOllama(
        base_url="http://localhost:11434",
        model=model,
        temperature=temperature,
        num_predict=num_predict,
        reasoning=False,
    )

def ask(prompt: str, **kwargs) -> str:
    llm = get_llm(**kwargs)
    resp = llm.invoke(prompt)
    print("Raw response from Ollama:", resp)
    print(f"Prompt: {prompt}\nResponse: {resp.content.strip()}")
    return resp.content.strip()
