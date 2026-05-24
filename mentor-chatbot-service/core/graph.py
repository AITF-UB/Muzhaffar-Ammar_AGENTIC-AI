import os
import json
from typing import Annotated, TypedDict, Dict, Any
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableConfig

from .prompts import SYSTEM_PROMPT_CHAT, SYSTEM_PROMPT_EVALUASI
from .config import settings

class GraphState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    mode: str
    atp: str
    mapel: str
    materi: str
    level: str
    emosi: str
    bacaan: str
    quiz_data: Dict[str, Any]

# Initialize LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=settings.OPENAI_API_KEY,
    streaming=True
)

async def chatbot_node(state: GraphState, config: RunnableConfig):
    mode = state.get("mode", "chat")
    messages = state["messages"]
    
    if mode == "chat":
        sys_prompt = SYSTEM_PROMPT_CHAT.format(
            atp=state.get("atp", ""),
            mapel=state.get("mapel", ""),
            materi=state.get("materi", ""),
            level=state.get("level", ""),
            emosi=state.get("emosi", "tidak diketahui"),
            bacaan=state.get("bacaan", "Tidak ada bahan bacaan spesifik.")
        )
    else:
        quiz_data_str = json.dumps(state.get("quiz_data", {}), indent=2)
        sys_prompt = SYSTEM_PROMPT_EVALUASI.format(
            atp=state.get("atp", ""),
            quiz_data=quiz_data_str
        )
        
    full_messages = [SystemMessage(content=sys_prompt)] + messages
    
    response = await llm.ainvoke(full_messages, config)
    return {"messages": [response]}

workflow = StateGraph(GraphState)
workflow.add_node("chatbot", chatbot_node)
workflow.add_edge(START, "chatbot")
workflow.add_edge("chatbot", END)
