"""
Endpoint seguro para entregar la API key de Gemini al frontend autenticado.

El frontend Angular conecta DIRECTAMENTE a Gemini Live API (igual que el ejemplo React),
eliminando la capa de proxy FastAPI que causaba doble latencia.

FastAPI conserva su rol para:
  - Autenticación JWT
  - Gestión de documentos
  - Historial de conversaciones
  - Este endpoint de token seguro

NUNCA exponer GEMINI_API_KEY en variables de entorno del frontend.
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
import os
import logging

from app.models.models import User
from app.api.v1.routes.auth_endpoints import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/voice", tags=["voice"])


class VoiceTokenResponse(BaseModel):
    api_key: str


@router.get("/token", response_model=VoiceTokenResponse)
async def get_voice_token(current_user: User = Depends(get_current_user)):
    """
    Devuelve la API key de Gemini al usuario autenticado.

    El frontend usa esta key para conectar DIRECTAMENTE a:
      wss://generativelanguage.googleapis.com/ws/...

    Esto elimina la capa de proxy y reduce la latencia a menos de 1 segundo.
    """
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        logger.error("GEMINI_API_KEY no configurada en .env")
        raise HTTPException(
            status_code=503,
            detail="Servicio de voz no disponible — configuración incompleta"
        )

    logger.info(f"Token de voz entregado a usuario {current_user.id}")
    return VoiceTokenResponse(api_key=api_key)