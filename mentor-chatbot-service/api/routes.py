import os
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, AIMessage
from core.graph import workflow
from .schemas import ChatRequest, ChatResponse, ChatResponseData, EvaluasiRequest
from core.config import settings

router = APIRouter()

# Compile the workflow statelessly (no checkpointer)
app_graph = workflow.compile()

def build_messages_from_history(history, new_message=None):
    messages = []
    for msg in history:
        if msg.role.lower() in ["user", "siswa"]:
            messages.append(HumanMessage(content=msg.teks))
        else:
            messages.append(AIMessage(content=msg.teks))
    if new_message:
        messages.append(HumanMessage(content=new_message))
    return messages

@router.post("/mentor/pesan", response_model=ChatResponse)
async def mentor_pesan(req: ChatRequest):
    # No config thread_id needed for stateless
    config = {} 
    
    messages = build_messages_from_history(req.history, req.pesan)
    
    state_input = {
        "messages": messages,
        "mode": "chat",
        "atp": req.atp,
        "mapel": req.elemen_label,
        "materi": req.materi,
        "level": req.level,
        "emosi": req.konteks.emosi if req.konteks else "tidak diketahui",
        "bacaan": req.konteks.bacaan if req.konteks else "Tidak ada bahan bacaan spesifik."
    }
    
    try:
        final_state = await app_graph.ainvoke(state_input, config=config)
        bot_msg = final_state["messages"][-1].content
        
        return ChatResponse(
            data=ChatResponseData(balasan=bot_msg, sesi_id=req.sesi_id),
            meta=None,
            error=None
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/mentor/pesan/stream")
async def mentor_pesan_stream(req: ChatRequest):
    config = {}
    messages = build_messages_from_history(req.history, req.pesan)
    
    state_input = {
        "messages": messages,
        "mode": "chat",
        "atp": req.atp,
        "mapel": req.elemen_label,
        "materi": req.materi,
        "level": req.level,
        "emosi": req.konteks.emosi if req.konteks else "tidak diketahui",
        "bacaan": req.konteks.bacaan if req.konteks else "Tidak ada bahan bacaan spesifik."
    }

    async def event_generator():
        try:
            async for event in app_graph.astream_events(state_input, config, version="v2"):
                if event["event"] == "on_chat_model_stream":
                    chunk = event["data"]["chunk"].content
                    if chunk:
                        chunk_clean = chunk.replace("\n", " ")
                        yield f"data: {chunk_clean}\n\n"
            
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: [ERROR] {str(e)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/mentor/evaluasi", response_model=ChatResponse)
async def mentor_evaluasi(req: EvaluasiRequest):
    config = {}
    messages = build_messages_from_history(req.history, "Tolong evaluasi hasil kuis saya.")
    
    state_input = {
        "messages": messages,
        "mode": "evaluasi",
        "atp": req.atp,
        "quiz_data": req.quiz_data or {}
    }
    
    try:
        final_state = await app_graph.ainvoke(state_input, config=config)
        bot_msg = final_state["messages"][-1].content
        
        return ChatResponse(
            data=ChatResponseData(balasan=bot_msg, sesi_id=req.sesi_id),
            meta=None,
            error=None
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/mentor/evaluasi/stream")
async def mentor_evaluasi_stream(req: EvaluasiRequest):
    config = {}
    messages = build_messages_from_history(req.history, "Tolong evaluasi hasil kuis saya.")
    
    state_input = {
        "messages": messages,
        "mode": "evaluasi",
        "atp": req.atp,
        "quiz_data": req.quiz_data or {}
    }

    async def event_generator():
        try:
            async for event in app_graph.astream_events(state_input, config, version="v2"):
                if event["event"] == "on_chat_model_stream":
                    chunk = event["data"]["chunk"].content
                    if chunk:
                        chunk_clean = chunk.replace("\n", " ")
                        yield f"data: {chunk_clean}\n\n"
            
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: [ERROR] {str(e)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
