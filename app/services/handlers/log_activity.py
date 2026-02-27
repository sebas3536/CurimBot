import logging
from .base import DocumentHandler, DocumentContext
from app.models.models import ActivityLog

class LogActivityHandler(DocumentHandler):
    """Handler para registrar actividades en la base de datos"""
    
    async def _handle(self, context: DocumentContext):
        """
        Registra la actividad de subida de documento
        """
        try:
            if not context.document:
                logging.warning(f"[{context.correlation_id}] No se puede registrar actividad: documento no creado")
                return
            
            # Obtener IP address del request si está disponible
            ip_address = None
            if hasattr(context, 'request') and context.request:
                ip_address = getattr(context.request.client, 'host', None) if context.request.client else None
            
            activity = ActivityLog(
                action="upload",
                document_id=context.document.id,
                user_id=context.user.id,
                document_name=context.document.filename,
                document_type=context.document.file_type,
                ip_address=ip_address
            )
            
            context.db.add(activity)
            context.db.commit()
            
            logging.info(
                f"[{context.correlation_id}] ✅ Actividad registrada: "
                f"Usuario {context.user.email} subió {context.document.filename} "
                f"(ID: {context.document.id})"
            )
            
        except Exception as e:
            logging.error(
                f"[{context.correlation_id}]  Error registrando actividad: {e}"
            )
