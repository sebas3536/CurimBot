"""
Router para carga de documentos con procesamiento en cadena (Chain of Responsibility).

Este módulo implementa un sistema robusto de carga de documentos con validación,
extracción de texto, encriptación opcional, guardado en BD e indexación en
Curim para búsqueda full-text. Usa un patrón Chain of Responsibility para
procesamiento modular y mantenible.

Características:
    - Validación de archivos (tamaño, tipo, contenido)
    - Extracción automática de texto (OCR-like)
    - Encriptación opcional de archivos
    - Guardado en base de datos
    - Indexación automática en CURIM para búsqueda
    - Registro de actividades de auditoría
    - Manejo robusto de errores
    - Correlación ID para rastreo de solicitudes
    - Procesamiento secuencial o asincrónico

Chain de Handlers:
    1. ValidateFileHandler: Valida archivo (tamaño, tipo, contenido)
    2. ExtractTextHandler: Extrae texto del documento
    3. [EncryptFileHandler]: Encripta archivo (opcional)
    4. SaveToDBHandler: Guarda en base de datos
    5. IndexCurimHandler: Indexa en CURIM para búsqueda
    6. LogActivityHandler: Registra actividad de auditoría
"""

import asyncio
import uuid
from fastapi import APIRouter, File, UploadFile, HTTPException, status, Depends
from typing import List
from pydantic import ValidationError

import logging

from requests import session
from app.db.crud import crud
from app.db.database import SessionLocal
from app.enums.enums import FileType
from app.models.models import CurimDocumentIndex, Document
from app.schemas.document_schemas import (
    DocumentCreate, DocumentOut, DocumentUploadResponse, UserDocumentSettings
)
from app.services.Curim.Curim_service import CurimService
from app.services.auth_service import get_current_user
from app.services.document_service import DocumentService
from app.services.handlers.base import DocumentContext, DocumentHandler, verify_chain_integrity
from app.services.handlers.encrypt_file import EncryptFileHandler
from app.services.handlers.extract_text import ExtractTextHandler
from app.services.handlers.log_activity import LogActivityHandler
from app.services.handlers.save_to_db import SaveToDBHandler
from app.services.handlers.validate_file import ValidateFileHandler
from app.services.handlers.index_Curim import IndexCurimHandler

router = APIRouter(prefix="/documents", tags=["documents"])
Curim_service = CurimService()


def get_db():
    """
    Obtener sesión de base de datos para usar en endpoints.
    
    Crea una nueva sesión SQLAlchemy y garantiza su cierre automático
    incluso si ocurre una excepción durante la solicitud.
    
    Yields:
        Session: Sesión de base de datos activa y disponible
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.post("/upload", response_model=DocumentUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_documents(
    files: List[UploadFile] = File(...),
    current_user=Depends(get_current_user),
    db: session=Depends(get_db)
):
    """
    Cargar múltiples documentos con validación, procesamiento e indexación.
    
    Este endpoint maneja la carga de documentos con un pipeline completo de
    procesamiento. Cada archivo pasa por una cadena de handlers que realizan
    validación, extracción de texto, encriptación opcional, guardado en BD,
    indexación en Curim e logging de actividad.
    
    Flujo de procesamiento (Chain of Responsibility):
        1. **ValidateFileHandler**: Valida archivo
           - Verifica tamaño (no vacío, no exceder límite)
           - Valida tipo de archivo (extensión, MIME type)
           - Verifica contenido (magic bytes)
        
        2. **ExtractTextHandler**: Extrae texto
           - PDF → OCR + extracción de texto
           - DOCX/XLSX → Extracción de contenido
           - TXT → Lectura directa
           - Resultado: texto completo en context.extracted_text
        
        3. **[EncryptFileHandler]**: Encripta (opcional)
           - Si user_settings.auto_encrypt = true
           - Usa Fernet para encriptación simétrica
           - Almacena clave de forma segura
        
        4. **SaveToDBHandler**: Guarda en base de datos
           - Crea registro en tabla Document
           - Almacena blob encriptado
           - Guarda metadatos (tamaño, tipo, hash)
           - Resultado: context.document con ID asignado
        
        5. **IndexCurimHandler**: Indexa en Curim
           - Chunifica el texto extraído
           - Envía chunks a Curim
           - Crea embeddings para búsqueda semántica
           - Guarda relación en CurimDocumentIndex
           - Resultado: documento indexado y searchable
        
        6. **LogActivityHandler**: Registra actividad
           - Crea registro de auditoría
           - Registra usuario, archivo, timestamp
           - Registra IP del cliente
           - Información para compliance
    
    Características de confiabilidad:
        - **Correlation ID**: Cada request tiene ID único para rastreo
        - **Logging detallado**: Logs en cada etapa del proceso
        - **Manejo de errores**: Fallo de un archivo no afecta otros
        - **Validación de cadena**: Verifica que no hay ciclos
        - **Procesamiento transaccional**: BD rollback en caso de error
    
    Límites y restricciones:
        - **Máximo archivos por request**: user_settings.max_file_types (default 10)
        - **Máximo tamaño por archivo**: user_settings.max_file_size (default 100MB)
        - **Tipos permitidos**: Según FileType enum (pdf, docx, xlsx, txt)
    
    Args:
        files (List[UploadFile]): Lista de archivos a cargar.
            - Mínimo 1 archivo
            - Máximo según user_settings
            - Formatos: pdf, docx, xlsx, txt
        
        current_user (User): Usuario autenticado (inyectado automáticamente)
        
        db (session): Sesión de base de datos (inyectada automáticamente)
    
    Returns:
        DocumentUploadResponse: Resumen de la carga incluyendo:
            - successful_uploads: Lista de DocumentOut (documentos creados exitosamente)
            - failed_uploads: Lista de errores por archivo
            - total_processed: Total de archivos procesados
            - success_count: Cantidad de éxitos
            - failure_count: Cantidad de fallos
            - Curim_indexed_count: Cantidad indexada en Curim (NUEVO)
    
    Raises:
        HTTPException 400: Sin archivos o demasiados archivos
        HTTPException 401: Usuario no autenticado
        HTTPException 413: Archivo excede tamaño máximo
        HTTPException 500: Error en procesamiento
    
    Example (éxito):
        POST /documents/upload
        Headers: Authorization: Bearer <access_token>
        Body: multipart/form-data con múltiples archivos
        
        Response (201 Created):
        {
            "successful_uploads": [
                {
                    "id": 1,
                    "name": "Reporte Q4.pdf",
                    "file_type": "pdf",
                    "size_bytes": 2097152,
                    "size_formatted": "2.0 MB",
                    "status": "completed",
                    "has_text_extracted": true,
                    "processing_progress": 100
                },
                {
                    "id": 2,
                    "name": "Presupuesto.xlsx",
                    "file_type": "xlsx",
                    "size_bytes": 524288,
                    "size_formatted": "512 KB",
                    "status": "completed",
                    "has_text_extracted": true,
                    "processing_progress": 100
                }
            ],
            "failed_uploads": [],
            "total_processed": 2,
            "success_count": 2,
            "failure_count": 0,
            "Curim_indexed_count": 2
        }
    
    Example (con fallos):
        POST /documents/upload
        Headers: Authorization: Bearer <access_token>
        Body: multipart/form-data
        
        Response (201 Created):
        {
            "successful_uploads": [
                {
                    "id": 1,
                    "name": "Reporte.pdf",
                    ...
                }
            ],
            "failed_uploads": [
                {
                    "filename": "huge_file.pdf",
                    "error": "Archivo demasiado grande (máximo 100MB)"
                },
                {
                    "filename": "corrupted.doc",
                    "error": "Formato de archivo no soportado"
                }
            ],
            "total_processed": 3,
            "success_count": 1,
            "failure_count": 2,
            "athelia_indexed_count": 1
        }
    
    Información sobre Curim:
        - **Indexación automática**: Cada documento se indexa en Curim
        - **Búsqueda full-text**: Permite búsquedas semánticas
        - **Chunificación**: Texto se divide en chunks para mejor búsqueda
        - **Embeddings**: Se generan vectores para búsqueda por similitud
        - **Conteo de chunks**: Se registra cantidad de chunks por documento
    
    Información en respuesta (Curim):
        - **athelia_indexed_count**: Cuántos documentos fueron indexados en Curim
        - Cada documento tiene flag "indexed_in_Curim" (si se agregara)
        - Indica éxito del pipeline completo
    
    Security:
        - Validación de tipo MIME para prevenir uploads maliciosos
        - Encriptación opcional de archivos
        - Verificación de tamaño para prevenir DoS
        - Logging completo para auditoría
        - Correlation ID para investigación
    
    Performance:
        - Procesamiento secuencial (más seguro, más lento)
        - Posible implementar asincrónico para archivos grandes
        - Curim indexación es relativamente rápida
        - Típicamente < 5 segundos para documentos pequeños
        - Puede requerir más tiempo para OCR de PDFs grandes
    
    Monitoring:
        - Correlation ID en todos los logs
        - Cada etapa se registra (inicio, fin, errores)
        - Resumen final incluye contador de indexaciones Curim
        - Errores detallados por archivo para debugging
    
    Best Practices:
        - Mostrar barra de progreso al usuario
        - Manejar errores por archivo sin detener otros
        - Guardar Correlation ID para soporte
        - Implementar reintentos para fallos transitorios
        - Monitorear logs de Curim para problemas
    
    Futuras mejoras:
        - Procesamiento asincrónico paralelo
        - Re-indexación en Curim si necesario
        - Estadísticas de éxito/fallo por tipo
        - Notificaciones de finalización
        - Webhooks para integración externa
    """
    # Generar ID único para rastrear esta solicitud
    correlation_id = str(uuid.uuid4())
    logging.info(f"[Correlation ID: {correlation_id}] Starting document upload for user {current_user.id}")
    logging.info(f"[Correlation ID: {correlation_id}] Number of files received: {len(files)}")

    # Obtener configuración del usuario
    user_settings = UserDocumentSettings()  # En práctica, obtener de BD

    # Validar entrada
    if not files:
        logging.error(f"[Correlation ID: {correlation_id}] No files provided")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="No se han enviado archivos"
        )

    if len(files) > user_settings.max_file_types:
        logging.error(f"[Correlation ID: {correlation_id}] Too many files: {len(files)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"No se permiten más de {user_settings.max_file_types} archivos por solicitud",
        )

    failed_files = []
    created_documents = []

    def build_handler_chain():
        """
        Construir cadena de handlers (Chain of Responsibility pattern).
        
        Orden de ejecución:
        1. ValidateFileHandler - Valida archivo
        2. ExtractTextHandler - Extrae texto
        3. [EncryptFileHandler] - Encripta (opcional)
        4. SaveToDBHandler - Guarda en BD
        5. IndexCurimHandler - Indexa en Curim ✅ NUEVO
        6. LogActivityHandler - Registra actividad
        
        Verifica que no hay ciclos en la cadena.
        """
        validate_handler = ValidateFileHandler()
        extract_handler = ExtractTextHandler()
        save_handler = SaveToDBHandler()
        Curim_handler = IndexCurimHandler()  
        log_handler = LogActivityHandler()
        
        # Asegurar que el último handler no tenga siguiente
        log_handler._next_handler = None
        
        # Construir cadena
        validate_handler.set_next(extract_handler)
        
        # Agregar encriptación si está habilitada
        if user_settings.auto_encrypt:
            encrypt_handler = EncryptFileHandler()
            extract_handler.set_next(encrypt_handler)
            encrypt_handler.set_next(save_handler)
        else:
            extract_handler.set_next(save_handler)
        
        # Agregar Curim después de guardar en BD
        # Esto asegura que el documento exista en BD antes de indexar
        save_handler.set_next(Curim_handler)
        Curim_handler.set_next(log_handler)
        
        # Verificar integridad (sin ciclos)
        if not verify_chain_integrity(validate_handler):
            raise RuntimeError("La cadena de handlers tiene un ciclo!")
        
        return validate_handler

    async def process_file(file: UploadFile, file_index: int):
        """
        Procesar un archivo individual a través de la cadena de handlers.
        
        Args:
            file: Archivo a procesar
            file_index: Índice del archivo en la lista
        
        Returns:
            Document si éxito, dict con error si fallo
        """
        try:
            logging.info(f"[Correlation ID: {correlation_id}] Processing file {file_index + 1}: {file.filename}")
            
            if not file.filename:
                return {"filename": "<sin nombre>", "error": "Nombre de archivo inválido"}

            # Leer contenido del archivo
            content = await file.read()
            size = len(content)
            
            logging.info(f"[Correlation ID: {correlation_id}] File {file.filename} - Size: {size} bytes")

            # Validar tamaño de archivo
            if size == 0 or size > user_settings.max_file_size:
                error_msg = "Archivo vacío" if size == 0 else "Archivo demasiado grande"
                logging.warning(f"[Correlation ID: {correlation_id}] File validation failed: {error_msg}")
                return {"filename": file.filename, "error": error_msg}

            # Crear contexto para pasar entre handlers
            context = DocumentContext(
                filename=file.filename,
                content=content,
                user=current_user,
                db=db,
                mimetype=file.content_type
            )
            context.size = size
            context.file = file
            context.correlation_id = correlation_id

            # IMPORTANTE: Crear cadena NUEVA para cada archivo
            # Esto evita problemas de estado compartido entre archivos
            handler_chain = build_handler_chain()
            
            logging.info(f"[Correlation ID: {correlation_id}] Starting handler chain for {file.filename}")

            # Procesar archivo a través de la cadena
            await handler_chain.handle(context)

            if not context.document:
                logging.error(f"[Correlation ID: {correlation_id}] Document not created for {file.filename}")
                return {"filename": file.filename, "error": "No se pudo guardar el documento"}

            logging.info(
                f"[Correlation ID: {correlation_id}] Successfully created document ID: {context.document.id} "
                f"for {file.filename}"
            )
            return context.document

        except HTTPException as he:
            logging.error(f"[Correlation ID: {correlation_id}] HTTP error processing file {file.filename}: {he.detail}")
            return {"filename": file.filename, "error": he.detail}
        except Exception as e:
            logging.exception(
                f"[Correlation ID: {correlation_id}] Unexpected error processing file {file.filename}: {str(e)}"
            )
            return {"filename": file.filename, "error": str(e)}
      
    # Procesamiento SECUENCIAL (recomendado para confiabilidad)
    for idx, file in enumerate(files):
        result = await process_file(file, idx)
        
        if isinstance(result, dict):
            # Diccionario significa error
            failed_files.append(result)
        elif isinstance(result, Document):
            # Documento significa éxito
            created_documents.append(result)
    
    # Validar y formatear respuesta
    valid_docs = []
    Curim_indexed_count = 0
    
    for doc in created_documents:
        try:
            # Verificar si fue indexado en Curim
            Curim_info = db.query(CurimDocumentIndex).filter(
                CurimDocumentIndex.document_id == doc.id
            ).first()
            
            # Agregar información de Curim al documento
            doc.Curim_indexed = Curim_info.is_indexed if Curim_info else False
            doc.Curim_chunks = Curim_info.chunks_count if Curim_info else 0
            
            # Contar documentos indexados
            if doc.Curim_indexed:
                Curim_indexed_count += 1
            
            # Convertir a esquema Pydantic
            valid_doc = DocumentOut.model_validate(doc)
            valid_docs.append(valid_doc)
            
        except ValidationError as ve:
            logging.error(f"[Correlation ID: {correlation_id}] Validation error: {ve.json()}")
            failed_files.append({
                "filename": doc.filename, 
                "error": "Error de validación de documento"
            })

    # Loguear resumen (incluyendo información de Curim)
    logging.info(
        f"[Correlation ID: {correlation_id}] Upload completed: "
        f"{len(valid_docs)} successful, {len(failed_files)} failed, "
        f"{Curim_indexed_count} indexed in Curim"  
    )

    # Retornar respuesta con información de Curim
    return DocumentUploadResponse(
        successful_uploads=valid_docs,
        failed_uploads=failed_files if failed_files else None,
        total_processed=len(files),
        success_count=len(valid_docs),
        failure_count=len(failed_files),
        Curim_indexed_count=Curim_indexed_count  
    )
