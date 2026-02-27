"""
VoiceRAGService — Wrapper del VoiceRAGEngine existente para el asistente de voz.

Usa el vectorstore Chroma con embeddings semánticos (HuggingFace multilingual)
que ya está indexado cuando el usuario sube documentos.

Flujo:
    1. Usuario habla → Gemini transcribe
    2. Frontend envía transcripción → POST /api/v1/voice/context
    3. VoiceRAGService busca en Chroma los chunks más relevantes del usuario
    4. Retorna contexto formateado para inyectar en Gemini Live
"""

import logging
from typing import Optional
from sqlalchemy.orm import Session

from app.models.models import Document
from app.services.Curim.rag_engine import VoiceRAGEngine

logger = logging.getLogger(__name__)

# Singleton del RAGEngine — se inicializa una sola vez al arrancar FastAPI
_rag_engine: Optional[VoiceRAGEngine] = None


def get_rag_engine() -> VoiceRAGEngine:
    """Retorna la instancia singleton del VoiceRAGEngine."""
    global _rag_engine
    if _rag_engine is None:
        logger.info("[VoiceRAG] Inicializando VoiceRAGEngine...")
        _rag_engine = VoiceRAGEngine()
        logger.info("[VoiceRAG] VoiceRAGEngine listo.")
    return _rag_engine


class VoiceRAGService:
    """
    Conecta el endpoint de voz con el VoiceRAGEngine existente.
    Recupera contexto semántico de los documentos del usuario
    para enriquecer las respuestas de Gemini Live.
    """

    def retrieve_context(
        self,
        query: str,
        user_id: int,
        db: Session,
        max_chars: int = 3000,
    ) -> Optional[str]:
        """
        Busca fragmentos semánticamente relevantes en los documentos del usuario.

        Usa el mismo vectorstore Chroma que ya se construye al subir documentos.
        Filtra automáticamente por document_id para que cada usuario
        solo vea sus propios documentos.

        Args:
            query:     Transcripción de lo que dijo el usuario.
            user_id:   ID del usuario autenticado.
            db:        Sesión de base de datos.
            max_chars: Límite de caracteres del contexto resultante.

        Returns:
            String con contexto formateado, o None si no hay matches.
        """
        if not query or not query.strip():
            return None

        # Obtener documentos del usuario que tengan texto extraído e indexado
        documents = (
            db.query(Document)
            .filter(
                Document.uploaded_by == user_id,
                Document.text.isnot(None),
                Document.text != "",
            )
            .order_by(Document.created_at.desc())
            .all()
        )

        if not documents:
            logger.info(f"[VoiceRAG] Usuario {user_id} no tiene documentos indexados.")
            return None

        logger.info(
            f"[VoiceRAG] Buscando en {len(documents)} documentos "
            f"para: '{query[:60]}'"
        )

        try:
            engine = get_rag_engine()

            # _retrieve_context_sync del RAGEngine existente:
            # - Hace similarity_search_with_score en Chroma
            # - Filtra por {"document_id": {"$in": doc_ids}}
            # - Retorna los 6 chunks más relevantes con score < 1.5
            context = engine._retrieve_context_sync(query, documents)

            # Respuestas vacías del engine
            NO_RESULT_RESPONSES = {
                "No hay documentos disponibles.",
                "No se encontró información relevante.",
                "No se encontraron chunks con suficiente relevancia.",
                "",
            }

            if not context or context in NO_RESULT_RESPONSES:
                logger.info(
                    f"[VoiceRAG] Sin matches semánticos para: '{query[:60]}'"
                )
                return None

            # Truncar si supera el límite para no sobrecargar Gemini
            if len(context) > max_chars:
                context = context[:max_chars] + "..."

            logger.info(
                f"[VoiceRAG] Contexto recuperado: {len(context)} chars "
                f"de {len(documents)} documento(s) disponibles"
            )
            return context

        except Exception as e:
            logger.error(f"[VoiceRAG] Error en retrieval: {e}", exc_info=True)
            # No propagar — la conversación debe continuar sin contexto
            return None


# Singleton del servicio
_voice_rag_service: Optional[VoiceRAGService] = None


def get_voice_rag_service() -> VoiceRAGService:
    global _voice_rag_service
    if _voice_rag_service is None:
        _voice_rag_service = VoiceRAGService()
    return _voice_rag_service