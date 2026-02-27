# app/services/handlers/encrypt_file.py
import logging
from app.services import storage_service
from .base import DocumentHandler, DocumentContext

class EncryptFileHandler(DocumentHandler):
    async def _handle(self, context: DocumentContext):
        try:
            if hasattr(context, 'content') and context.content:
                context.encrypted_content = storage_service.encrypt_bytes(context.content)
            else:
                context.encrypted_content = b""
        except Exception as e:
            logging.exception(f"Error encriptando archivo: {e}")
            context.encrypted_content = b""
