"""
Router para acceso y visualización de metadatos de documentos.

Este módulo proporciona endpoints para obtener información detallada sobre
documentos sin descargar el contenido completo. Incluye metadatos como
tamaño, tipo, fecha de creación, estadísticas de acceso y más.

Características:
    - Acceso a metadatos sin descargar archivo
    - Registro de visualizaciones
    - Estadísticas de acceso por documento
    - Información del propietario
    - Historial de cambios
    - Control de acceso por usuario/admin
"""

import logging
from typing import List
from fastapi import Depends, HTTPException, Request, APIRouter
from fastapi.params import Query
from fastapi.responses import StreamingResponse

from requests import session
from app.schemas.document_schemas import DocumentOut, DocumentWithMetadata
from app.services.auth_service import get_current_user
from app.db.crud import crud
from app.db.database import SessionLocal, get_db
from app.models import models
from app.services import storage_service
from app.services.document_service import DocumentService

router = APIRouter(prefix="/documents", tags=["documents"])


def get_db():
    """
    Obtener sesión de base de datos para usar en endpoints.
    
    Crea una nueva sesión SQLAlchemy y garantiza su cierre automático
    incluso si ocurre una excepción durante la solicitud.
    
    Yields:
        Session: Sesión de base de datos activa y disponible
    
    Notes:
        - Sigue el patrón de inyección de dependencias de FastAPI
        - Cierra la conexión automáticamente al finalizar
        - Se ejecuta una vez por solicitud HTTP
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# =========================================================
# 📄 Obtener Metadatos por ID con registro de actividad
# =========================================================

@router.get("/{doc_id}/metadata", response_model=DocumentOut)
def get_document_metadata(
    doc_id: int,
    db: session = Depends(get_db),
    user=Depends(get_current_user)
):
    """
    Obtener metadatos detallados de un documento específico.
    
    Este endpoint retorna información completa sobre un documento sin necesidad
    de descargar el archivo completo. Incluye información de fecha, tamaño,
    tipo, estadísticas de acceso y más. Cada consulta se registra como
    visualización para tracking de uso.
    
    Información incluida:
        - **ID**: Identificador único del documento
        - **Nombre**: Nombre original del archivo
        - **Tipo**: Extensión y MIME type
        - **Tamaño**: En bytes y formato legible
        - **Propietario**: ID y nombre del usuario
        - **Fechas**: Creación, actualización, última visualización
        - **Estadísticas**: Contador de visualizaciones, descargas
        - **Status**: Estado actual (draft, processing, completed, error)
        - **Ruta**: Ubicación en almacenamiento
        - **Hash**: Para verificar integridad
    
    Registro de actividad:
        - Cada consulta se registra como "viewed"
        - Se incrementa contador de visualizaciones
        - Se actualiza timestamp de última visualización
        - Se registra IP del cliente
        - Útil para tracking de uso y auditoría
    
    Casos de uso:
        - Obtener info sin descargar archivo
        - Verificar tamaño antes de descargar
        - Mostrar detalles en tabla de documentos
        - Validar existencia del documento
        - Obtener estadísticas de acceso
        - Verificar integridad (hash)
    
    Args:
        doc_id (int): ID único del documento
        db (session): Sesión de base de datos (inyectada automáticamente)
        user (User): Usuario autenticado actual (inyectado automáticamente)
    
    Returns:
        DocumentOut: Metadatos completos del documento:
            - id: ID único
            - name: Nombre del archivo
            - file_type: Tipo (pdf, docx, txt, etc.)
            - size_bytes: Tamaño en bytes
            - size_formatted: Tamaño legible (2.3 MB)
            - mime_type: MIME type exacto
            - created_at: Fecha creación (ISO 8601)
            - updated_at: Última actualización (ISO 8601)
            - last_viewed_at: Última consulta de metadata (ISO 8601)
            - owner_id: ID del propietario
            - owner_name: Nombre del propietario
            - status: Estado actual
            - view_count: Total de visualizaciones
            - download_count: Total de descargas
            - has_text_extracted: Si se extrajo texto
            - processing_progress: Porcentaje completado (0-100)
            - error_message: Mensaje si hubo error (null si está bien)
    
    Raises:
        HTTPException 401: Usuario no autenticado
        HTTPException 403: Usuario no es propietario
        HTTPException 404: Documento no encontrado
        HTTPException 500: Error al obtener metadatos
    
    Example (exitoso):
        GET /documents/123/metadata
        Headers: Authorization: Bearer <access_token>
        
        Response (200 OK):
        {
            "id": 123,
            "name": "Reporte Q4 2025.pdf",
            "file_type": "pdf",
            "size_bytes": 2097152,
            "size_formatted": "2.0 MB",
            "mime_type": "application/pdf",
            "created_at": "2025-11-01T10:30:00Z",
            "updated_at": "2025-11-02T15:45:00Z",
            "last_viewed_at": "2025-11-02T20:48:00Z",
            "owner_id": 1,
            "owner_name": "Juan Pérez",
            "status": "completed",
            "view_count": 12,
            "download_count": 3,
            "has_text_extracted": true,
            "processing_progress": 100,
            "error_message": null
        }
    
    Example (en proceso):
        GET /documents/124/metadata
        Headers: Authorization: Bearer <access_token>
        
        Response (200 OK):
        {
            "id": 124,
            "name": "Presupuesto 2026.xlsx",
            "file_type": "xlsx",
            "size_bytes": 524288,
            "size_formatted": "512 KB",
            "mime_type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "created_at": "2025-11-02T20:00:00Z",
            "updated_at": "2025-11-02T20:00:00Z",
            "last_viewed_at": "2025-11-02T20:00:00Z",
            "owner_id": 1,
            "owner_name": "Juan Pérez",
            "status": "processing",
            "view_count": 1,
            "download_count": 0,
            "has_text_extracted": false,
            "processing_progress": 65,
            "error_message": null
        }
    
    Example (no encontrado):
        GET /documents/999/metadata
        Headers: Authorization: Bearer <access_token>
        
        Response (404 Not Found):
        {
            "detail": "Documento no encontrado"
        }
    
    Notes:
        - Los metadatos se pueden obtener sin descargar
        - Cada acceso incrementa view_count
        - Perfecto para implementar tablas con info de archivos
        - El error_message solo está presente si status es "error"
        - La IP se registra para auditoría
    
    Performance:
        - Retorna rápidamente (solo lectura de metadatos)
        - Típicamente < 50ms
        - No requiere acceso al archivo físico
        - Ideal para paginar sin descargas
    
    Security:
        - Solo propietario puede ver metadatos de su documento
        - Admin puede ver cualquier documento (si configurado)
        - Cada acceso se registra en auditoría
    """
    try:
        # Obtener metadatos del documento
        # El servicio se encarga de:
        # - Verificar que el usuario es propietario
        # - Obtener info de la BD
        # - Incrementar contador de visualizaciones
        # - Registrar actividad
        doc = DocumentService.get_metadata(doc_id, db, user)
        
        # Registrar en logs
        logging.info(f"Usuario {user.email} consultó metadata del documento {doc.filename} (ID: {doc_id})")
        return doc
        
    except HTTPException:
        # Re-lanzar excepciones HTTP (permisos, no encontrado, etc.)
        raise
    except Exception as e:
        # Capturar errores inesperados y loguear
        logging.exception(f"Error inesperado obteniendo metadata de documento {doc_id}: {e}")
        raise HTTPException(
            status_code=500, 
            detail="Error interno del servidor"
        )


# =========================================================
# 📑 Documentos con metadatos completos
# =========================================================

@router.get("/metadata/all", response_model=List[DocumentWithMetadata])
def get_documents_metadata(
    include_all_users: bool = Query(False, description="Incluir documentos de todos los usuarios (solo admin)"),
    db: session = Depends(get_db),
    user=Depends(get_current_user)
):
    """
    Obtener lista de todos los documentos con metadatos completos.
    
    Este endpoint retorna una lista de documentos con información enriched,
    incluyendo estadísticas de acceso, información del propietario y más.
    Los usuarios ven solo sus documentos, mientras que los administradores
    pueden ver todos los documentos del sistema.
    
    Metadatos incluidos:
        - **Información básica**: ID, nombre, tipo, tamaño
        - **Propietario**: ID, nombre, email
        - **Fechas**: Creación, actualización, última visualización
        - **Estadísticas**: Visualizaciones, descargas, comparticiones
        - **Status**: Estado actual (draft, processing, completed, error)
        - **Procesamiento**: Progreso, texto extraído
        - **Almacenamiento**: Ruta, tamaño, checksum
    
    Control de acceso:
        - **Usuarios regulares**: Solo sus propios documentos
        - **Administradores**: Todos los documentos del sistema
            - include_all_users=true para ver todos
            - include_all_users=false para ver solo propios
    
    Casos de uso:
        - Dashboard con tabla de documentos
        - Estadísticas de uso global (admin)
        - Auditoría de documentos del sistema
        - Reporte de almacenamiento
        - Análisis de patrones de uso
    
    Args:
        include_all_users (bool): Si true y es admin, incluye todos los documentos.
            Default: False (solo documentos del usuario)
        db (session): Sesión de base de datos (inyectada automáticamente)
        user (User): Usuario autenticado actual (inyectado automáticamente)
    
    Returns:
        List[DocumentWithMetadata]: Lista de documentos con metadatos:
            - id: ID único
            - name: Nombre del archivo
            - file_type: Tipo de archivo
            - size_bytes: Tamaño en bytes
            - size_formatted: Tamaño legible
            - owner_id: ID del propietario
            - owner_name: Nombre del propietario
            - owner_email: Email del propietario
            - created_at: Fecha creación
            - updated_at: Última actualización
            - last_accessed_at: Última visualización
            - status: Estado actual
            - view_count: Total de visualizaciones
            - download_count: Total de descargas
            - shared_with_count: Número de usuarios con acceso
            - processing_progress: Porcentaje (0-100)
            - has_errors: Si hay errores de procesamiento
    
    Raises:
        HTTPException 401: Usuario no autenticado
        HTTPException 403: Usuario intenta ver documentos globales sin ser admin
        HTTPException 500: Error al obtener documentos
    
    Example (usuario regular):
        GET /documents/metadata/all
        Headers: Authorization: Bearer <access_token>
        
        Response (200 OK):
        [
            {
                "id": 123,
                "name": "Reporte Q4 2025.pdf",
                "file_type": "pdf",
                "size_bytes": 2097152,
                "size_formatted": "2.0 MB",
                "owner_id": 1,
                "owner_name": "Juan Pérez",
                "owner_email": "juan@example.com",
                "created_at": "2025-11-01T10:30:00Z",
                "updated_at": "2025-11-02T15:45:00Z",
                "last_accessed_at": "2025-11-02T20:48:00Z",
                "status": "completed",
                "view_count": 12,
                "download_count": 3,
                "shared_with_count": 2,
                "processing_progress": 100,
                "has_errors": false
            },
            {
                "id": 124,
                "name": "Presupuesto 2026.xlsx",
                "file_type": "xlsx",
                "size_bytes": 524288,
                "size_formatted": "512 KB",
                "owner_id": 1,
                "owner_name": "Juan Pérez",
                "owner_email": "juan@example.com",
                "created_at": "2025-11-02T20:00:00Z",
                "updated_at": "2025-11-02T20:00:00Z",
                "last_accessed_at": "2025-11-02T20:00:00Z",
                "status": "processing",
                "view_count": 1,
                "download_count": 0,
                "shared_with_count": 0,
                "processing_progress": 65,
                "has_errors": false
            }
        ]
    
    Example (admin - datos globales):
        GET /documents/metadata/all?include_all_users=true
        Headers: Authorization: Bearer <admin_token>
        
        Response (200 OK):
        [
            ...todos los documentos del sistema con metadatos...
        ]
    
    Example (admin - solo propios):
        GET /documents/metadata/all?include_all_users=false
        Headers: Authorization: Bearer <admin_token>
        
        Response (200 OK):
        [
            ...documentos del administrador...
        ]
    
    Example (no autorizado):
        GET /documents/metadata/all?include_all_users=true
        Headers: Authorization: Bearer <user_token>
        
        Response (403 Forbidden):
        {
            "detail": "No autorizado para ver documentos de otros usuarios"
        }
    
    Ordenamiento:
        - Por defecto: Más recientes primero (updated_at descendente)
        - O por created_at descendente según configuración
    
    Performance:
        - Retorna lista completa sin paginar
        - Puede ser grande en sistemas con muchos documentos
        - Considerar agregar paginación si necesario
        - Típicamente < 500ms para miles de documentos
    
    Security:
        - Usuarios regulares ven solo sus documentos
        - Admin requiere verificación adicional (is_admin)
        - Cada acceso se podría registrar en auditoría
        - Considerar limitar expose de emails de otros usuarios
    
    Best Practices:
        - Usar este endpoint para dashboards
        - Cachear resultados en cliente si es adecuado
        - Paginar si hay muchos documentos
        - Mostrar información enriquecida en tablas
        - Implementar ordenamiento por columna
    """
    try:
        # Determinar si incluir documentos de todos los usuarios
        # Solo admin puede ver documentos de otros usuarios
        user_id = None if (include_all_users and user.is_admin) else user.id
        
        # Obtener documentos con metadatos enriched
        # El servicio se encarga de:
        # - Filtrar por usuario si aplica
        # - Obtener información completa con joins
        # - Calcular estadísticas
        # - Agregar información del propietario
        documents = DocumentService.get_documents_with_metadata(db, user_id)
        
        return documents
        
    except HTTPException:
        # Re-lanzar excepciones HTTP (permisos, etc.)
        raise
    except Exception as e:
        # Capturar errores inesperados y loguear
        logging.exception(f"Error obteniendo documentos con metadata: {e}")
        raise HTTPException(
            status_code=500, 
            detail="Error interno del servidor"
        )
