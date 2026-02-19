"""
WebSocket route for live chat using the RAG service.
"""

import asyncio
from typing import Any, Dict

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.api.service import RAGApiService

router = APIRouter(tags=["chat"])


def _extract_text(payload: Dict[str, Any]) -> str:
    return str(payload.get("message") or payload.get("text") or "").strip()


@router.websocket("/ws/chat")
async def ws_chat(websocket: WebSocket) -> None:
    await websocket.accept()
    service: RAGApiService = websocket.app.state.rag_service
    active_chat_id = service.create_session()
    await websocket.send_json({"type": "ready", "chat_id": active_chat_id})

    try:
        while True:
            data = await websocket.receive_json()
            action = str(data.get("type", "message")).lower()

            if action in {"close", "disconnect"}:
                await websocket.send_json({"type": "bye"})
                break
            if action == "new_chat":
                title = str(data.get("title") or "").strip() or None
                chat = service.create_chat(title=title)
                active_chat_id = chat["chat_id"]
                await websocket.send_json({"type": "new_chat_ok", "chat": chat, "active_chat_id": active_chat_id})
                continue
            if action == "switch_chat":
                requested = str(data.get("chat_id") or "").strip()
                if not requested:
                    await websocket.send_json({"type": "error", "error": "chat_id is required"})
                    continue
                service.get_chat(requested)
                active_chat_id = requested
                await websocket.send_json({"type": "switch_chat_ok", "active_chat_id": active_chat_id})
                continue
            if action == "reset":
                service.reset_session(active_chat_id)
                await websocket.send_json({"type": "reset_ok", "chat_id": active_chat_id})
                continue

            message = _extract_text(data)
            if not message:
                await websocket.send_json({"type": "error", "error": "Empty message"})
                continue

            await websocket.send_json({"type": "processing"})
            result = await asyncio.to_thread(
                service.chat,
                active_chat_id,
                message,
                True,
            )
            await websocket.send_json(
                {
                    "type": "answer",
                    "chat_id": active_chat_id,
                    "answer": result.get("answer", ""),
                    "sources": result.get("sources", []),
                }
            )
    except WebSocketDisconnect:
        pass
    finally:
        service.close_session(active_chat_id)
