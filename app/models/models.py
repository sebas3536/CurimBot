"""
Módulo de modelos ORM para base de datos.

Define todas las tablas y relaciones de la base de datos usando SQLAlchemy ORM.
Incluye modelos para autenticación, documentos, actividades, preferencias,
sesiones, recuperación de contraseña y Curim.

Estructura:
    - Mixins: TimestampMixin para created_at/updated_at
    - Autenticación: User, Role, LoginAttempt, LoginAlert
    - Documentos: Document, ActivityLog
    - Sesiones: ActiveSession, BlacklistedToken
    - Recuperación: PasswordResetToken
    - Preferencias: UserPreferences
    - Curim: Conversaciones, mensajes, indexación
"""

from typing import Optional
from sqlalchemy import (
    Boolean, Column, Integer, String, Text, LargeBinary, DateTime,
    ForeignKey, Index, CheckConstraint, UniqueConstraint
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
from sqlalchemy import Enum as SqlEnum
from sqlalchemy.dialects.mysql import LONGBLOB, LONGTEXT
from app.db.database import Base
from app.enums.enums import FileType, LogAction


# =========================================================
# MIXINS - Campos comunes
# =========================================================

class TimestampMixin:
    """
    Mixin para agregar campos de timestamp automáticos.
    
    Proporciona created_at y updated_at a cualquier modelo que lo herede.
    Los timestamps se generan automáticamente por la BD.
    
    Campos:
        - created_at: Cuándo se creó el registro (inmutable)
        - updated_at: Cuándo se actualizó por última vez
    
    Uso:
        class MiTabla(Base, TimestampMixin):
            __tablename__ = "mi_tabla"
            # ... otros campos ...
    """
    created_at = Column(
        DateTime, 
        server_default=func.now(), 
        nullable=False
    )
    updated_at = Column(
        DateTime, 
        server_default=func.now(), 
        onupdate=func.now(), 
        nullable=False
    )


# =========================================================
# AUTENTICACIÓN
# =========================================================

class Role(Base):
    """
    Modelo de roles para control de acceso.
    
    Define los roles disponibles en el sistema (admin, user, etc).
    Se usa para autorización de endpoints y control granular de permisos.
    
    Campos:
        - id: Identificador único
        - name: Nombre único del rol (admin, user)
        - description: Descripción legible
    
    Relaciones:
        - users: Usuarios con este rol
    
    Ejemplo de datos:
        - Role(name='admin', description='Administrator with full system access')
        - Role(name='user', description='Regular user with standard permissions')
    """
    __tablename__ = "roles"

    id = Column(Integer, primary_key=True)
    name = Column(
        String(50), 
        unique=True, 
        nullable=False,
        index=True
    )
    description = Column(Text, nullable=True)

    users = relationship("User", back_populates="role")

    def __repr__(self):
        return f"<Role(id={self.id}, name={self.name})>"


class User(Base, TimestampMixin):
    """
    Modelo de usuario principal.
    
    Tabla central con toda la información del usuario incluyendo
    autenticación, 2FA, sesiones activas, y referencias a otros recursos.
    
    Campos principales:
        - id: Identificador único
        - email: Correo único (usado para login)
        - password_hash: Hash de contraseña (nunca guardar en claro)
        - name: Nombre completo
        - is_active: Si la cuenta está activa
        - last_login: Último login exitoso
        - role_id: FK a Role
    
    Campos de seguridad:
        - failed_attempts: Intentos fallidos de login (para bloqueo)
        - locked_until: Hasta cuándo está bloqueada la cuenta
    
    Campos 2FA:
        - two_factor_enabled: Si 2FA está activado
        - two_factor_secret: Secreto TOTP (32 caracteres)
        - two_factor_secret_temp: Secreto temporal durante setup
        - backup_codes: Códigos de respaldo (separados, probablemente JSON)
    
    Relaciones:
        - role: Role del usuario
        - documents: Documentos subidos
        - logs: Logs del usuario
        - activities: Actividades registradas
        - preferences: Preferencias personales (uno-a-uno)
        - login_alerts: Alertas de login sospechosas
        - active_sessions: Sesiones activas en dispositivos
        - reset_tokens: Tokens de recuperación de contraseña
    
    Propiedades:
        - is_admin: True si role.name == 'admin'
        - is_user: True si role.name == 'user'
    """
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    password_hash = Column(String(128), nullable=False)
    name = Column(String(100), nullable=False)

    is_active = Column(Boolean, default=True, nullable=False)
    last_login = Column(DateTime, nullable=True)

    # FK a Role
    role_id = Column(Integer, ForeignKey("roles.id"), nullable=False)
    role = relationship("Role", back_populates="users")
    
    # Seguridad: bloqueo de cuenta tras intentos fallidos
    failed_attempts = Column(Integer, default=0, nullable=False)
    locked_until = Column(DateTime, nullable=True)

    # Autenticación de dos factores
    two_factor_enabled = Column(Boolean, default=False, nullable=False)
    two_factor_secret = Column(String(64), nullable=True)
    two_factor_secret_temp = Column(String(64), nullable=True)
    two_factor_enabled_at = Column(DateTime, nullable=True)
    two_factor_disabled_at = Column(DateTime, nullable=True)
    backup_codes = Column(Text, nullable=True)

    # Relaciones
    documents = relationship(
        "Document", 
        back_populates="owner", 
        cascade="all, delete-orphan"
    )
    logs = relationship("Log", back_populates="user")
    activities = relationship("ActivityLog", back_populates="user")
    preferences = relationship(
        "UserPreferences", 
        back_populates="user", 
        uselist=False, 
        cascade="all, delete-orphan"
    )
    login_alerts = relationship(
        "LoginAlert", 
        back_populates="user", 
        cascade="all, delete-orphan"
    )
    active_sessions = relationship(
        "ActiveSession", 
        back_populates="user", 
        cascade="all, delete-orphan"
    )
    reset_tokens = relationship(
        "PasswordResetToken", 
        back_populates="user", 
        cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<User(id={self.id}, email={self.email}, role={self.role.name})>"

    @property
    def is_admin(self) -> bool:
        """Verifica si el usuario es administrador"""
        return self.role and self.role.name == "admin"
    
    @property
    def is_user(self) -> bool:
        """Verifica si el usuario es regular"""
        return self.role and self.role.name == "user"


class LoginAttempt(Base):
    """
    Registro de intentos de login.
    
    Almacena cada intento de login (exitoso o fallido) para:
    - Rate limiting (bloquear tras N intentos fallidos)
    - Auditoría (quién intentó acceder, cuándo, desde dónde)
    - Detección de fuerza bruta
    
    Campos:
        - id: Identificador único
        - email: Email que intentó loguear
        - ip_address: IP del cliente (IPv4 o IPv6)
        - success: True si login fue exitoso
        - attempted_at: Cuándo fue el intento
        - user_agent: Navegador/cliente
    
    Notas:
        - Se registra incluso si email no existe (para seguridad)
        - Se usa para bloqueo temporal de cuenta
        - Se debe limpiar periódicamente (después de N días)
    """
    __tablename__ = "login_attempts"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), index=True, nullable=False)
    ip_address = Column(String(45), nullable=True)  # IPv6 puede ser más largo
    success = Column(Boolean, default=False, nullable=False)
    attempted_at = Column(DateTime, server_default=func.now(), nullable=False)
    user_agent = Column(String(255), nullable=True)


class LoginAlert(Base):
    """
    Alertas de login sospechoso.
    
    Detecta y registra logins desde nuevos dispositivos o ubicaciones,
    permitiendo notificar al usuario de posibles accesos no autorizados.
    
    Campos:
        - device: Descripción del dispositivo (ej: 'Chrome 120 on Windows')
        - location: Ubicación geográfica si disponible
        - ip_address: IP del cliente
        - is_suspicious: Marcado como potencialmente sospechoso
        - is_new_device: Primer login desde este dispositivo
        - is_new_location: Primer login desde esta ubicación
        - notification_sent: Si se envió notificación al usuario
    
    Relaciones:
        - user: Usuario al que pertenece el alert
    """
    __tablename__ = "login_alerts"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    
    # Información del login
    device = Column(String(200), nullable=False)
    location = Column(String(200), nullable=True)
    ip_address = Column(String(45), nullable=False)
    user_agent = Column(Text, nullable=True)
    
    # Banderas de sospecha
    is_suspicious = Column(Boolean, default=False, nullable=False)
    is_new_device = Column(Boolean, default=False, nullable=False)
    is_new_location = Column(Boolean, default=False, nullable=False)
    
    # Notificación
    notification_sent = Column(Boolean, default=False, nullable=False)
    notification_sent_at = Column(DateTime, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    user = relationship("User", back_populates="login_alerts")

    def __repr__(self):
        return f"<LoginAlert(user_id={self.user_id}, device={self.device}, suspicious={self.is_suspicious})>"


# =========================================================
# DOCUMENTOS
# =========================================================

class Document(Base, TimestampMixin):
    """
    Modelo de documento.
    
    Almacena documentos subidos por usuarios. Incluye metadatos,
    contenido encriptado, texto extraído y contadores de acceso.
    
    Campos principales:
        - id: Identificador único
        - filename: Nombre del archivo original
        - mimetype: Tipo MIME (application/pdf, etc)
        - size: Tamaño en bytes
        - file_type: Tipo de archivo (enum)
        - uploaded_by: FK a User propietario
    
    Campos de contenido:
        - text: Texto extraído del documento
        - blob_enc: Contenido binario encriptado
        - encryption_version: Versión de encriptación usada
    
    Contadores:
        - view_count: Cuántas veces se visualizó
        - download_count: Cuántas veces se descargó
        - last_accessed: Última vez que se accedió
    
    Metadatos:
        - status: Estado (uploaded, processing, completed, error)
    
    Índices:
        - (uploaded_by, file_type): Para filtrado rápido
        - Constraint: size >= 0
    """
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), index=True, nullable=False)
    mimetype = Column(String(100), nullable=False)
    size = Column(Integer, default=0, nullable=False)
    file_type = Column(
        SqlEnum(FileType, name="file_type_enum"), 
        nullable=False
    )
    text = Column(LONGTEXT, nullable=True)  
    blob_enc = Column(LONGBLOB, nullable=True) 
    encryption_version = Column(Integer, default=1, nullable=False)

    # Propietario
    uploaded_by = Column(Integer, ForeignKey("users.id"), nullable=False)
    owner = relationship("User", back_populates="documents")

    # Contadores de acceso
    last_accessed = Column(DateTime, nullable=True)
    download_count = Column(Integer, default=0, nullable=False)
    view_count = Column(Integer, default=0, nullable=False)

    # Relaciones
    activities = relationship(
        "ActivityLog", 
        back_populates="document", 
        passive_deletes=True
    )

    Curim_index = relationship(
        "CurimDocumentIndex",
        back_populates="document",
        passive_deletes=True,  
        uselist=False, 
        cascade="all, delete-orphan"
    )
    
    # Estado del documento
    status = Column(String(20), default="uploaded", nullable=False)
    
    # Índices y constraints
    __table_args__ = (
        Index('idx_uploaded_by_file_type', 'uploaded_by', 'file_type'),
        CheckConstraint('size >= 0', name='check_document_size_positive')
    )
    
    @property
    def uploaded_by_name(self) -> Optional[str]:
        """Nombre del usuario que subió el documento"""
        return self.owner.name if self.owner else None


class ActivityLog(Base):
    """
    Registro de actividades sobre documentos.
    
    Auditoría de todas las acciones del usuario sobre documentos.
    Permite rastrear quién accedió, descargó, compartió, eliminó, etc.
    
    Campos:
        - action: Tipo de acción (view, download, upload, delete, share)
        - document_id: Documento afectado
        - user_id: Usuario que realizó la acción
        - timestamp: Cuándo ocurrió
        - ip_address: IP del cliente
        - document_name: Nombre del documento (snapshot)
        - document_type: Tipo del documento (snapshot)
    
    Relaciones:
        - user: Usuario que realizó la acción
        - document: Documento afectado (puede ser null si fue eliminado)
    """
    __tablename__ = "activity_logs"

    id = Column(Integer, primary_key=True, index=True)
    action = Column(String(20), nullable=False)
    document_id = Column(
        Integer, 
        ForeignKey("documents.id", ondelete="SET NULL")
    )
    user_id = Column(Integer, ForeignKey("users.id"))
    timestamp = Column(
        DateTime, 
        default=datetime.utcnow, 
        server_default=func.now(), 
        nullable=False
    )
    ip_address = Column(String(45), nullable=True)
    document_name = Column(String(255), nullable=True)
    document_type = Column(
        SqlEnum(FileType, name="activity_file_type_enum")
    )

    user = relationship("User", back_populates="activities")
    document = relationship(
        "Document", 
        back_populates="activities", 
        passive_deletes=True
    )


class BlacklistedToken(Base):
    """
    Tokens revocados (logout).
    
    Almacena tokens que han sido explícitamente revocados
    (usuario hizo logout). Se verifica para invalidar tokens
    durante validación en endpoints protegidos.
    
    Campos:
        - token: Token JWT revocado (string)
        - blacklisted_at: Cuándo se revocó
    """
    __tablename__ = "blacklisted_tokens"

    id = Column(Integer, primary_key=True, index=True)
    token = Column(String(512), unique=True, index=True, nullable=False)
    blacklisted_at = Column(DateTime, server_default=func.now(), nullable=False)


# =========================================================
# PREFERENCIAS Y CONFIGURACIÓN
# =========================================================

class UserPreferences(Base):
    """
    Preferencias personalizadas del usuario.
    
    Almacena configuración del usuario como idioma, tema, 
    preferencias de notificación, etc.
    
    Campos de notificación:
        - email_notifications: Recibir emails
        - push_notifications: Notificaciones push
        - weekly_summary: Resumen semanal
        - login_alerts: Alertas de login sospechoso
    
    Campos de interfaz:
        - language: Idioma (es, en, etc)
        - theme: Tema visual (light, dark, auto)
        - profile_photo_url: URL de foto de perfil
    
    Relaciones:
        - user: Usuario propietario (uno-a-uno)
    """
    __tablename__ = "user_preferences"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(
        Integer, 
        ForeignKey("users.id", ondelete="CASCADE"), 
        unique=True, 
        nullable=False
    )
    
    # Preferencias de notificación
    email_notifications = Column(Boolean, default=True, nullable=False)
    push_notifications = Column(Boolean, default=False, nullable=False)
    weekly_summary = Column(Boolean, default=True, nullable=False)
    login_alerts = Column(Boolean, default=True, nullable=False)
    
    # Preferencias de interfaz
    language = Column(String(5), default="es", nullable=False)
    theme = Column(String(10), default="light", nullable=False)
    
    # Foto de perfil
    profile_photo_url = Column(String(500), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relación con User
    user = relationship("User", back_populates="preferences")
    
    def __repr__(self):
        return f"<UserPreferences(user_id={self.user_id}, lang={self.language}, theme={self.theme})>"


# =========================================================
# SESIONES Y RECUPERACIÓN
# =========================================================

class ActiveSession(Base):
    """
    Sesiones activas del usuario.
    
    Rastrea sesiones activas en diferentes dispositivos permitiendo:
    - Gestión de múltiples dispositivos
    - Revocación de sesiones específicas
    - Vista de dispositivos activos
    - Logout de todos los dispositivos
    
    Campos:
        - access_token_jti: JWT ID del access token
        - refresh_token_jti: JWT ID del refresh token
        - device: Descripción del dispositivo
        - ip_address: IP del cliente
        - location: Ubicación geográfica
        - created_at: Cuándo se creó la sesión
        - last_active: Último uso
        - expires_at: Cuándo expira (basado en refresh token)
        - is_current: Si es la sesión actual del request
        - is_active: Si está activa o revocada
    """
    __tablename__ = "active_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(
        Integer, 
        ForeignKey("users.id", ondelete="CASCADE"), 
        nullable=False
    )
    
    # JWT IDs
    access_token_jti = Column(
        String(255), 
        unique=True, 
        nullable=False, 
        index=True
    )
    refresh_token_jti = Column(
        String(255), 
        unique=True, 
        nullable=False, 
        index=True
    )
    
    # Información del dispositivo
    device = Column(String(255), nullable=True)
    ip_address = Column(String(50), nullable=True)
    user_agent = Column(Text, nullable=True)
    location = Column(String(255), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_active = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    expires_at = Column(DateTime, nullable=False)
    
    # Estado
    is_current = Column(Boolean, default=False, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    
    # Relación
    user = relationship("User", back_populates="active_sessions")
    
    def __repr__(self):
        return f"<ActiveSession(user_id={self.user_id}, device={self.device}, active={self.is_active})>"


class PasswordResetToken(Base):
    """
    Tokens para recuperación de contraseña.
    
    Tokens temporales para permitir reset de contraseña.
    Se envía por email al usuario. De un solo uso.
    
    Campos:
        - token: Token único (string largo)
        - expires_at: Cuándo expira (típicamente 1 hora)
        - is_used: Si ya fue usado
        - used_at: Cuándo se usó
    
    Relaciones:
        - user: Usuario que solicitó reset
    """
    __tablename__ = "password_reset_tokens"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(
        Integer, 
        ForeignKey("users.id", ondelete="CASCADE"), 
        nullable=False
    )
    token = Column(String(255), unique=True, nullable=False, index=True)
    expires_at = Column(DateTime, nullable=False)
    is_used = Column(Boolean, default=False, nullable=False)
    used_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    user = relationship("User", back_populates="reset_tokens")


class Log(Base):
    """
    Logs del sistema para auditoría.
    
    Registra eventos importantes del sistema para auditoría y debugging.
    
    Campos:
        - user_id: Usuario afectado
        - action: Tipo de acción (enum LogAction)
        - detail: Detalles adicionales
        - created_at: Cuándo ocurrió
    """
    __tablename__ = "logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    action = Column(
        SqlEnum(LogAction, name="log_action_enum"), 
        nullable=False, 
        index=True
    )
    detail = Column(Text, nullable=True)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)

    user = relationship("User", back_populates="logs")

    def __repr__(self):
        return f"<Log(id={self.id}, user_id={self.user_id}, action={self.action})>"


# =========================================================
# Curim - CONVERSACIONES Y INDEXACIÓN
# =========================================================

class CurimConversation(Base, TimestampMixin):
    """
    Conversaciones con el asistente Curim.
    
    Una conversación agrupa múltiples mensajes de un usuario
    con Curim (asistente de IA).
    
    Campos:
        - user_id: Usuario propietario
        - title: Título auto-generado de la conversación
        - is_active: Si está activa o archivada
    
    Relaciones:
        - messages: Mensajes de la conversación
    """
    __tablename__ = "Curim_conversations"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(
        Integer, 
        ForeignKey("users.id", ondelete="CASCADE"), 
        nullable=False, 
        index=True
    )
    title = Column(String(255), nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    
    user = relationship("User", backref="Curim_conversations")
    messages = relationship(
        "CurimMessage", 
        back_populates="conversation", 
        cascade="all, delete-orphan"
    )
    
    def __repr__(self):
        return f"<CurimConversation(id={self.id}, user_id={self.user_id})>"


class CurimMessage(Base):
    """
    Mensajes individuales de una conversación.
    
    Cada mensaje representa una entrada o salida de la conversación.
    
    Campos:
        - conversation_id: Conversación a la que pertenece
        - role: 'user' o 'assistant'
        - content: Texto del mensaje
        - confidence: Confianza de respuesta (0-100)
        - sources: Documentos usados para generar respuesta (JSON)
        - from_cache: Si la respuesta vino de caché
        - processing_time_ms: Tiempo de procesamiento
    """
    __tablename__ = "Curim_messages"
    
    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(
        Integer, 
        ForeignKey("Curim_conversations.id", ondelete="CASCADE"), 
        nullable=False, 
        index=True
    )
    role = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    
    confidence = Column(Integer, nullable=True)
    sources = Column(Text, nullable=True)
    from_cache = Column(Boolean, default=False, nullable=False)
    processing_time_ms = Column(Integer, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    conversation = relationship("CurimConversation", back_populates="messages")
    
    __table_args__ = (
        Index('idx_conversation_created', 'conversation_id', 'created_at'),
    )
    
    def __repr__(self):
        return f"<CurimMessage(id={self.id}, role={self.role})>"


class CurimDocumentIndex(Base, TimestampMixin):
    """
    Tracking de documentos indexados en Curim.
    
    Registra qué documentos están indexados en Curim
    para búsqueda semántica y generación de respuestas.
    
    Campos:
        - document_id: Documento indexado
        - is_indexed: Si está correctamente indexado
        - index_version: Versión del índice
        - chunks_count: Cantidad de chunks generados
        - last_indexed_at: Cuándo se indexó por última vez
        - error_message: Mensaje de error si falló
    """
    __tablename__ = "Curim_document_index"
    
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(
        Integer, 
        ForeignKey("documents.id", ondelete="CASCADE"), 
        nullable=False, 
        unique=True, 
        index=True
    )
    is_indexed = Column(Boolean, default=False, nullable=False)
    index_version = Column(Integer, default=1, nullable=False)
    chunks_count = Column(Integer, default=0, nullable=False)
    last_indexed_at = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)
    
    document = relationship(
        "Document",
        back_populates="Curim_index"
    )
    
    def __repr__(self):
        return f"<CurimDocumentIndex(doc={self.document_id}, indexed={self.is_indexed})>"