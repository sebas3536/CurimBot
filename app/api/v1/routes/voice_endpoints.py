"""
Endpoints para el asistente de voz.

GET  /api/v1/voice/token    — API key de Gemini para conexión directa
POST /api/v1/voice/context  — Fragmentos RAG relevantes para la query
"""

import logging
import os

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.services.Curim.voice_rag_service import get_voice_rag_service
from app.db.database import SessionLocal
from app.services.auth_service import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/voice", tags=["voice"])


# ─────────────────────────────────────────────────────────────────────────────
# Dependencias
# ─────────────────────────────────────────────────────────────────────────────

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ─────────────────────────────────────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────────────────────────────────────

class VoiceTokenResponse(BaseModel):
    api_key: str


class ContextRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000,
                       description="Transcripción de lo que dijo el usuario")


class ContextResponse(BaseModel):
    context: str | None = Field(
        None,
        description="Fragmentos relevantes de documentos, o null si no hay matches"
    )
    found: bool = Field(
        False,
        description="True si se encontró contexto relevante"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/token", response_model=VoiceTokenResponse)
async def get_voice_token(current_user=Depends(get_current_user)):
    """
    Devuelve la API key de Gemini al usuario autenticado.
    El frontend la usa para conectar directamente a Gemini Live API.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY no configurada")
        raise HTTPException(
            status_code=503,
            detail="Servicio de voz no disponible"
        )
    return VoiceTokenResponse(api_key=api_key)


@router.post("/context", response_model=ContextResponse)
async def get_voice_context(
    request: ContextRequest,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Busca fragmentos relevantes en los documentos del usuario
    para enriquecer la respuesta del asistente de voz.

    El frontend llama este endpoint cada vez que Gemini transcribe
    lo que dijo el usuario, antes de que Gemini procese la respuesta.
    """
    try:
        rag = get_voice_rag_service()
        context = rag.retrieve_context(
            query=request.query,
            user_id=current_user.id,
            db=db,
        )

        return ContextResponse(
            context=context,
            found=context is not None,
        )

    except Exception as e:
        logger.error(f"[RAG] Error recuperando contexto: {e}", exc_info=True)
        # No fallar la conversación por error de RAG
        return ContextResponse(context=None, found=False)