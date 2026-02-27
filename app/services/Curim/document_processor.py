"""
Procesador de documentos para Curim.

Maneja extracción de texto, validación, chunificación e indexación
de documentos en la base de datos vectorial para búsqueda semántica.

Flow:
    1. Documento se sube (con texto extraído)
    2. DocumentProcessor.validate_document() verifica
    3. DocumentProcessor.process_and_index() chunifica e indexa
    4. Se crea CurimDocumentIndex en BD
    5. Documento es searchable en Curim

Supported formats:
    - PDF (with OCR)
    - DOCX (Word)
    - DOC (Word legacy)
    - TXT (plain text)

Typical usage:
    processor = DocumentProcessor()
    
    # Validate document first (recommended)
    if processor.validate_document(doc):
        chunks = processor.process_and_index(doc)
        print(f"Indexed {chunks} chunks")
    else:
        print("Document cannot be processed")
"""

import logging
from typing import Optional, List, Dict, Any
from enum import Enum
from pathlib import Path

from app.models.models import Document
from app.services.Curim.rag_engine import VoiceRAGEngine as RAGEngine

# Configurar logger
logger = logging.getLogger(__name__)


class ValidationResult:
    """Resultado de validación con detalles."""
    
    def __init__(self, is_valid: bool, reason: Optional[str] = None):
        self.is_valid = is_valid
        self.reason = reason
    
    def __bool__(self) -> bool:
        return self.is_valid
    
    def __str__(self) -> str:
        if self.is_valid:
            return "Valid document"
        return f"Invalid document: {self.reason}"

class DocumentProcessor:
    """
    Procesador de documentos para indexación en Curim.
    
    Responsable de tomar documentos con texto extraído y procesarlos
    para indexación en el vectorstore. Valida documentos antes de
    procesamiento y maneja errores gracefully.
    
    Attributes:
        rag_engine: Motor de búsqueda vectorial
        min_text_length: Longitud mínima de texto para procesar
        supported_types: Tipos de archivo soportados
    
    Configuration:
        - MIN_TEXT_LENGTH: 50 caracteres
        - SUPPORTED_TYPES: ['pdf', 'txt', 'docx', 'doc']
    
    Example:
        >>> processor = DocumentProcessor()
        >>> doc = Document(id=123, filename="test.pdf", text="...")
        >>> if processor.validate_document(doc):
        ...     chunks = processor.process_and_index(doc)
        ...     print(f"Document indexed in {chunks} chunks")
    """
    
    # Constantes de configuración
    MIN_TEXT_LENGTH = 50
    SUPPORTED_TYPES = ['pdf', 'txt', 'docx', 'doc']
    
    def __init__(self, rag_engine: Optional[RAGEngine] = None):
        """
        Inicializar procesador.
        
        Args:
            rag_engine: Instancia de RAGEngine (opcional)
                       Si no se provee, crea una nueva
        
        Example:
            # With default RAGEngine
            processor = DocumentProcessor()
            
            # With custom RAGEngine
            rag = RAGEngine(config={"chunk_size": 1000})
            processor = DocumentProcessor(rag_engine=rag)
        """
        self.rag_engine = rag_engine or RAGEngine()
        logger.info("DocumentProcessor initialized")
    
    def process_and_index(self, document: Document) -> int:
        """
        Procesar documento e indexar en vectorstore.
        
        Toma un documento con texto ya extraído, lo chunifica
        en segmentos manejables, crea embeddings, e indexa
        en la base de datos vectorial para búsqueda semántica.
        
        Args:
            document: Modelo de documento con texto extraído
                - document.id: ID único
                - document.text: Texto extraído del archivo
                - document.filename: Nombre original para referencia
        
        Returns:
            Número de chunks creados y indexados
        
        Raises:
            ValueError: Si documento no tiene texto extraído
            IndexError: Si falla la indexación en vectorstore
            Exception: Para otros errores durante el procesamiento
        
        Example:
            try:
                chunks = processor.process_and_index(document)
                logger.info(f"Document {document.id} indexed with {chunks} chunks")
            except ValueError as e:
                logger.error(f"Invalid document: {e}")
            except Exception as e:
                logger.error(f"Indexing failed: {e}")
        
        Notes:
            - Documento debe tener texto extraído previamente
            - El embedding se hace con Gemini API
            - Chunks se persisten en vectorstore (Chroma)
            - Si el documento ya existe, se actualiza (upsert)
        """
        # Validar documento antes de procesar
        validation = self.validate_document(document, raise_on_invalid=False)
        if not validation:
            raise ValueError(
                f"Cannot index document {document.id}: {validation.reason}"
            )
        
        try:
            # Indexar en el vectorstore
            # RAGEngine maneja chunificación, embeddings y almacenamiento
            chunks_count = self.rag_engine.index_document(document)
            
            logger.info(
                f"Document {document.id} ({document.filename}) "
                f"indexed successfully with {chunks_count} chunks"
            )
            
            return chunks_count
            
        except Exception as e:
            logger.error(
                f"Failed to index document {document.id}: {str(e)}",
                exc_info=True
            )
            raise
    
    def validate_document(
        self, 
        document: Document, 
        raise_on_invalid: bool = False
    ) -> ValidationResult:
        """
        Validar que un documento sea procesable.
        
        Realiza validaciones completas para asegurar que el documento
        puede ser procesado e indexado correctamente.
        
        Args:
            document: Modelo de documento a validar
            raise_on_invalid: Si True, lanza excepción en lugar de retornar False
        
        Returns:
            ValidationResult con estado y razón si es inválido
        
        Raises:
            ValueError: Si raise_on_invalid=True y documento es inválido
        
        Example:
            # Using boolean result
            result = processor.validate_document(doc)
            if result:
                print("Document is valid")
            else:
                print(f"Document invalid: {result.reason}")
            
            # Using exception mode
            try:
                processor.validate_document(doc, raise_on_invalid=True)
                print("Document is valid")
            except ValueError as e:
                print(f"Validation failed: {e}")
        
        Validations performed:
            1. Document has text content
            2. Text meets minimum length requirement
            3. File type is supported
            4. Document is not corrupted
        """
        try:
            # Validación 1: Verificar que tiene texto
            if not document.text:
                reason = "Document has no text extracted"
                return self._handle_validation_result(False, reason, raise_on_invalid)
            
            # Validación 2: Verificar longitud mínima
            text_length = len(document.text.strip())
            if text_length < self.MIN_TEXT_LENGTH:
                reason = (
                    f"Document text too short: {text_length} chars. "
                    f"Minimum required: {self.MIN_TEXT_LENGTH}"
                )
                return self._handle_validation_result(False, reason, raise_on_invalid)
            
            # Validación 3: Verificar tipo de archivo soportado
            if document.file_type.value not in self.SUPPORTED_TYPES:
                reason = (
                    f"Unsupported file type: {document.file_type.value}. "
                    f"Supported types: {', '.join(self.SUPPORTED_TYPES)}"
                )
                return self._handle_validation_result(False, reason, raise_on_invalid)
            
            # Validación 4: Verificar que el texto sea válido (no solo espacios/símbolos)
            if not self._has_meaningful_content(document.text):
                reason = "Document text lacks meaningful content (only whitespace/symbols)"
                return self._handle_validation_result(False, reason, raise_on_invalid)
            
            logger.debug(f"Document {document.id} passed validation")
            return ValidationResult(True)
            
        except Exception as e:
            reason = f"Validation error: {str(e)}"
            logger.error(f"Document validation failed: {reason}", exc_info=True)
            return self._handle_validation_result(False, reason, raise_on_invalid)
    
    def _handle_validation_result(
        self, 
        is_valid: bool, 
        reason: str, 
        raise_on_invalid: bool
    ) -> ValidationResult:
        """Manejar resultado de validación según configuración."""
        if not is_valid and raise_on_invalid:
            raise ValueError(reason)
        
        if not is_valid:
            logger.warning(f"Document validation failed: {reason}")
        
        return ValidationResult(is_valid, reason)
    
    def _has_meaningful_content(self, text: str) -> bool:
        """
        Verificar si el texto tiene contenido significativo.
        
        Args:
            text: Texto a verificar
        
        Returns:
            True si el texto tiene contenido significativo
        """
        # Remover espacios y ver si queda algo más que símbolos
        stripped = text.strip()
        
        # Verificar que no sea solo símbolos
        alpha_count = sum(1 for c in stripped if c.isalpha())
        return alpha_count > 0
    
    def bulk_process(
        self, 
        documents: List[Document], 
        continue_on_error: bool = False
    ) -> Dict[str, Any]:
        """
        Procesar múltiples documentos en lote.
        
        Args:
            documents: Lista de documentos a procesar
            continue_on_error: Si True, continúa con siguiente documento en error
        
        Returns:
            Dict con estadísticas del procesamiento:
                - total: Total de documentos procesados
                - successful: Documentos procesados exitosamente
                - failed: Documentos que fallaron
                - total_chunks: Total de chunks creados
                - errors: Lista de errores (si continue_on_error=True)
        
        Example:
            docs = [doc1, doc2, doc3]
            result = processor.bulk_process(docs, continue_on_error=True)
            
            print(f"Processed: {result['successful']}/{result['total']}")
            print(f"Total chunks: {result['total_chunks']}")
            
            for error in result['errors']:
                print(f"Error with doc {error['doc_id']}: {error['error']}")
        """
        results = {
            "total": len(documents),
            "successful": 0,
            "failed": 0,
            "total_chunks": 0,
            "errors": []
        }
        
        for doc in documents:
            try:
                chunks = self.process_and_index(doc)
                results["successful"] += 1
                results["total_chunks"] += chunks
                
            except Exception as e:
                results["failed"] += 1
                
                if continue_on_error:
                    results["errors"].append({
                        "doc_id": doc.id,
                        "filename": doc.filename,
                        "error": str(e)
                    })
                    logger.warning(f"Bulk processing: skipped document {doc.id}: {e}")
                else:
                    logger.error(f"Bulk processing failed at document {doc.id}")
                    raise
        
        logger.info(
            f"Bulk processing completed: "
            f"{results['successful']}/{results['total']} successful, "
            f"{results['total_chunks']} total chunks"
        )
        
        return results
    
    def get_processing_stats(self, document: Document) -> Dict[str, Any]:
        """
        Obtener estadísticas de procesamiento para un documento.
        
        Args:
            document: Documento a analizar
        
        Returns:
            Dict con estadísticas:
                - text_length: Longitud del texto
                - estimated_chunks: Chunks estimados
                - file_type: Tipo de archivo
                - is_processable: Si es procesable
                - validation_reason: Razón si no es procesable
        """
        validation = self.validate_document(document)
        
        stats = {
            "document_id": document.id,
            "filename": document.filename,
            "file_type": document.file_type.value,
            "text_length": len(document.text) if document.text else 0,
            "is_processable": validation.is_valid,
            "estimated_chunks": 0
        }
        
        if not validation.is_valid:
            stats["validation_reason"] = validation.reason
        
        # Estimar chunks (asumiendo chunk_size=500, overlap=100)
        if document.text and validation.is_valid:
            text_length = len(document.text)
            chunk_size = 500
            overlap = 100
            stats["estimated_chunks"] = max(
                1, 
                (text_length - overlap) // (chunk_size - overlap)
            )
        
        return stats