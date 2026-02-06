from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from app.prompts import SYSTEM_PROMPT

def build_llm(model_name: str):
    return ChatGroq(
        model=model_name,
        temperature=0.2,
        max_tokens=250,
    )

def llm_answer(llm, user_query: str, context_blocks: list[str]) -> str:
    context_text = "\n\n".join(context_blocks).strip()

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"""
CONTEXT (FAQ Knowledge):
{context_text}

USER QUESTION:
{user_query}

Task:
Answer ONLY using the CONTEXT. If not present, ask 1 short clarifying question.
""".strip())
    ]

    resp = llm.invoke(messages)
    return resp.content.strip()