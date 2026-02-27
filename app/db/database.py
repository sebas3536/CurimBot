"""
Módulo de configuración de base de datos MySQL.

Este módulo gestiona toda la configuración de conexión a MySQL,
incluyendo creación del motor SQLAlchemy, sesiones, modelos base y la
inyección de dependencias para FastAPI.

Configuración soportada:
    - MySQL local: Desarrollo y producción local
    - MySQL remoto: Producción en servidores externos

Componentes:
    - engine: Motor SQLAlchemy de conexión
    - SessionLocal: Factory de sesiones
    - Base: Declarative base para modelos ORM
    - get_db: Dependency injection para FastAPI
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
import os
from dotenv import load_dotenv

# Cargar variables de entorno desde .env
load_dotenv()

# =========================================================
#  CONFIGURACIÓN DE CONEXIÓN MYSQL
# =========================================================

# **DATABASE_URL**: URL de conexión a MySQL
#   
#   Formato: mysql+pymysql://usuario:contraseña@host:puerto/nombre_bd?charset=utf8mb4
#   
#   Ejemplos:
#   - MySQL local:
#     mysql+pymysql://root:password@localhost:3306/asistente_docs?charset=utf8mb4
#   
#   - MySQL remoto (producción):
#     mysql+pymysql://usuario:password@192.168.1.100:3306/asistente_docs?charset=utf8mb4
#     mysql+pymysql://usuario:password@db.ejemplo.com:3306/asistente_docs?charset=utf8mb4
#   
#   Parámetros importantes:
#   - charset=utf8mb4: Soporte completo UTF-8 (emojis, caracteres especiales)
#   - pymysql: Driver Python puro (sin dependencias C)
#   
#   Variables de entorno requeridas en .env:
#   DATABASE_URL=mysql+pymysql://usuario:password@host:puerto/nombre_bd?charset=utf8mb4
#   
#   O configurar por partes:
#   DB_USER=root
#   DB_PASSWORD=tu_password
#   DB_HOST=localhost  # o IP remota: 192.168.1.100
#   DB_PORT=3306
#   DB_NAME=asistente_docs

# Opción 1: URL completa desde variable de entorno
DATABASE_URL = os.getenv("DATABASE_URL")

# Opción 2: Construir URL desde variables individuales (si DATABASE_URL no existe)
if not DATABASE_URL:
    DB_USER = os.getenv("DB_USER", "root")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "Sebas_torres16")
    DB_HOST = os.getenv("DB_HOST", "localhost")  # Cambiar a IP remota si es necesario
    DB_PORT = os.getenv("DB_PORT", "3306")
    DB_NAME = os.getenv("DB_NAME", "curim_db")
    
    DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}?charset=utf8mb4"

# =========================================================
#  CREACIÓN DEL MOTOR SQLALCHEMY PARA MYSQL
# =========================================================

# **engine**: Motor SQLAlchemy optimizado para MySQL
#   - Pool de conexiones configurado para alta concurrencia
#   - Reconexión automática en caso de pérdida de conexión
#   - Configuración específica para MySQL
engine = create_engine(
    DATABASE_URL,
    # Pool de conexiones - ajustar según carga esperada
    pool_size=10,              # Conexiones permanentes en el pool
    max_overflow=20,           # Conexiones adicionales bajo alta carga
    pool_recycle=3600,         # Reciclar conexiones cada hora (evita "MySQL server has gone away")
    pool_pre_ping=True,        # Verificar conexión antes de usar (reconexión automática)
    
    # Configuración específica de MySQL
    connect_args={
        "charset": "utf8mb4",           # Soporte UTF-8 completo
        "use_unicode": True,            # Manejo correcto de Unicode
        "connect_timeout": 10,          # Timeout de conexión (10 segundos)
        "read_timeout": 30,             # Timeout de lectura (30 segundos)
        "write_timeout": 30,            # Timeout de escritura (30 segundos)
    },
    
    # Logging de queries (desactivar en producción)
    echo=False,  # True para ver queries SQL en consola (solo desarrollo)
)

# =========================================================
#  SESIONES DE BASE DE DATOS
# =========================================================

# **SessionLocal**: Factory para crear sesiones de BD MySQL
#   - Crea una nueva sesión para cada request
#   - autocommit=False: Requiere explicit commit (mejor control de transacciones)
#   - autoflush=False: No flushea automáticamente (mejor para debugging)
#   - bind=engine: Usa el motor MySQL definido arriba
SessionLocal = sessionmaker(
    autocommit=False,  # Transacciones explícitas (más control)
    autoflush=False,   # Flush explícito (mejor para debugging)
    bind=engine        # Usar nuestro motor MySQL
)

# **Base**: Base declarativa para definir modelos ORM
#   - Todos los modelos heredan de Base
#   - Base.metadata contiene definición de todas las tablas
#   - Se usa para crear_all() en inicialización
Base = declarative_base()

# =========================================================
#  DEPENDENCY INJECTION PARA FASTAPI
# =========================================================

def get_db():
    """
    Obtener sesión de MySQL para inyectar en endpoints.
    
    Esta función implementa el patrón de inyección de dependencias de FastAPI.
    Se usa para proporcionar sesiones de MySQL a los endpoints de forma automática.
    
    Características:
        - Crea nueva sesión para cada request
        - Cierra sesión automáticamente después
        - Maneja excepciones (cierra incluso si hay error)
        - Compatible con async/await
        - Optimizado para MySQL con pool de conexiones
    
    Uso en endpoints:
        ```python
        from fastapi import Depends
        from app.db.database import get_db
        from sqlalchemy.orm import Session
        
        @app.get("/users/")
        def get_users(db: Session = Depends(get_db)):
            # db es sesión MySQL automáticamente inyectada
            users = db.query(User).all()
            return users
        ```
    
    Ciclo de vida:
        1. FastAPI llama get_db() al recibir request
        2. Se obtiene conexión del pool MySQL
        3. Se crea sesión con SessionLocal()
        4. Se yield sesión al endpoint
        5. Endpoint usa la sesión
        6. Se ejecuta bloque finally
        7. Sesión se cierra (conexión vuelve al pool)
        8. Request completa
    
    Transacciones en MySQL:
        - Cada sesión es transacción independiente
        - Changes se guardan con db.commit()
        - Rollback automático en excepciones
        - MySQL usa InnoDB (soporte completo de transacciones ACID)
        - Nivel de aislamiento por defecto: REPEATABLE READ
    
    Pool de conexiones MySQL:
        - SessionLocal reutiliza conexiones del pool
        - pool_size=10: 10 conexiones permanentes
        - max_overflow=20: hasta 30 conexiones totales bajo carga
        - pool_recycle=3600: recicla cada hora (evita timeouts MySQL)
        - pool_pre_ping=True: verifica antes de usar (reconexión automática)
    
    Performance:
        - Sesión nueva por request (overhead mínimo)
        - Pool de conexiones reutiliza conexiones MySQL
        - Típicamente < 1ms por request desde pool
        - Primera conexión: 10-50ms (establecer conexión TCP)
    
    Error handling:
        - try/finally asegura cierre incluso con excepciones
        - pool_pre_ping reconecta si MySQL cerró la conexión
        - Si endpoint genera error, sesión se cierra igual
        - Previene connection leaks
        - MySQL "server has gone away" se maneja automáticamente
    
    Manejo de "MySQL server has gone away":
        - pool_recycle=3600: recicla conexiones antes del timeout MySQL
        - pool_pre_ping=True: verifica conexión antes de usar
        - Si conexión muerta, se obtiene nueva del pool automáticamente
    
    Testing:
        - Fácil de mockear para tests
        - Se puede inyectar sesión de test
        - Permite transacciones de test aisladas
        - Compatible con pytest-mysql
    
    Yields:
        Session: Sesión SQLAlchemy MySQL lista para usar
    
    Example:
        ```python
        # En endpoint
        @app.post("/documents/")
        def create_doc(doc_data: DocumentCreate, db: Session = Depends(get_db)):
            db_doc = Document(**doc_data.dict())
            db.add(db_doc)
            db.commit()
            db.refresh(db_doc)
            return db_doc
        
        # Ciclo con MySQL:
        # 1. get_db() obtiene conexión del pool
        # 2. endpoint usa sesión
        # 3. db.commit() persiste en MySQL
        # 4. finally db.close() devuelve conexión al pool
        ```
    
    Configuración remota:
        Para conectar a MySQL remoto, configurar en .env:
        ```
        DATABASE_URL=mysql+pymysql://usuario:password@IP_REMOTA:3306/nombre_bd?charset=utf8mb4
        ```
        
        Ejemplo producción:
        ```
        DATABASE_URL=mysql+pymysql://app_user:SecurePass123@192.168.1.100:3306/asistente_docs?charset=utf8mb4
        ```
        
        Consideraciones seguridad:
        - Usar usuario MySQL con permisos limitados (no root)
        - Password fuerte en producción
        - Firewall: solo permitir IP del servidor FastAPI
        - SSL/TLS recomendado: agregar ?ssl=true al DATABASE_URL
    """
    # Obtener sesión del pool MySQL
    db = SessionLocal()
    try:
        # Yield sesión al endpoint (es generador)
        yield db
    finally:
        # Cerrar sesión siempre, incluso si hay error
        # La conexión vuelve al pool (no se cierra realmente)
        db.close()


# =========================================================
#  INICIALIZACIÓN DE BASE DE DATOS MYSQL
# =========================================================

def init_db():
    """
    Inicializar base de datos MySQL creando todas las tablas.
    
    Ejecuta CREATE TABLE para todos los modelos registrados.
    Usar en primera ejecución o después de cambios en modelos.
    
    IMPORTANTE: 
    - No ejecutar en producción con datos existentes
    - Usar Alembic para migraciones en producción
    - Esta función no elimina datos existentes (safe)
    
    Uso:
        ```python
        from app.db.database import init_db
        
        # En main.py al iniciar la aplicación
        init_db()
        ```
    
    Notas MySQL:
        - Usa InnoDB como motor por defecto (transaccional)
        - Charset: utf8mb4 (soporte completo Unicode)
        - Collation: utf8mb4_unicode_ci (case-insensitive)
        - Los índices se crean automáticamente
        - Las foreign keys se crean con ON DELETE según modelo
    """
    # Importar modelos AQUÍ para evitar circular imports
    from app.models import models
    Base.metadata.create_all(bind=engine)


# =========================================================
#  REFERENCIA DE MODELOS REGISTRADOS EN MYSQL
# =========================================================

"""
Los siguientes modelos están registrados y se crearán en MySQL:

AUTENTICACIÓN:
1. **Role**: Roles de usuario
   - InnoDB, utf8mb4
   - Índice único en 'name'

2. **User**: Usuarios del sistema
   - InnoDB, utf8mb4
   - Índice único en 'email'
   - Índice en 'role_id'
   - Soporte para 2FA

3. **LoginAttempt**: Intentos de login
   - InnoDB, utf8mb4
   - Índice en 'email'
   - Para rate limiting

4. **LoginAlert**: Alertas de login sospechoso
   - InnoDB, utf8mb4
   - Índice en 'user_id'

DOCUMENTOS:
5. **Document**: Documentos del sistema
   - InnoDB, utf8mb4
   - Campo LONGBLOB para archivos grandes (hasta 4GB en MySQL)
   - TEXT para contenido extraído
   - Índice compuesto (uploaded_by, file_type)
   - Contadores de acceso (view_count, download_count)

6. **ActivityLog**: Auditoría de acciones
   - InnoDB, utf8mb4
   - Índice en (document_id, user_id)
   - ON DELETE SET NULL para documentos

SESIONES Y TOKENS:
7. **ActiveSession**: Sesiones activas
   - InnoDB, utf8mb4
   - Índice único en JWT IDs
   - Para multi-dispositivo

8. **BlacklistedToken**: Tokens revocados
   - InnoDB, utf8mb4
   - Índice único en 'token'

9. **PasswordResetToken**: Recuperación de contraseña
   - InnoDB, utf8mb4
   - Índice único en 'token'

PREFERENCIAS:
10. **UserPreferences**: Configuración de usuario
    - InnoDB, utf8mb4
    - Relación uno-a-uno con User
    - Índice único en 'user_id'

LOGS:
11. **Log**: Logs del sistema
    - InnoDB, utf8mb4
    - Índice en 'action'
    - Campo TEXT para detalles

Curim (IA):
12. **CurimConversation**: Conversaciones con IA
    - InnoDB, utf8mb4
    - Índice en 'user_id'

13. **CurimMessage**: Mensajes de conversaciones
    - InnoDB, utf8mb4
    - Índice compuesto (conversation_id, created_at)
    - Campo TEXT para contenido

14. **CurimDocumentIndex**: Índice de documentos para IA
    - InnoDB, utf8mb4
    - Índice único en 'document_id'

CONVOCATORIAS:
15. **Convocatoria**: Convocatorias principales
    - InnoDB, utf8mb4
    - Índice en 'created_by'

16. **ConvocatoriaDocument**: Documentos requeridos
    - InnoDB, utf8mb4
    - Índice en 'convocatoria_id'

17. **ConvocatoriaHistory**: Historial de cambios
    - InnoDB, utf8mb4
    - Índice en 'convocatoria_id'

18. **ConvocatoriaCollaborator**: Colaboradores
    - InnoDB, utf8mb4
    - Constraint único (convocatoria_id, user_id)

19. **ConvocatoriaGuideDocument**: Documento guía
    - InnoDB, utf8mb4
    - Relación uno-a-uno con Convocatoria

Todas las tablas se crean con:
    Base.metadata.create_all(bind=engine)

Características MySQL utilizadas:
- Motor InnoDB: Soporte transaccional ACID
- Charset utf8mb4: Soporte completo Unicode (emojis, etc)
- Collation utf8mb4_unicode_ci: Búsquedas case-insensitive
- Foreign Keys con ON DELETE CASCADE/SET NULL
- Índices optimizados para consultas frecuentes
- LONGBLOB para archivos grandes (hasta 4GB)
- TEXT para contenido largo
- DATETIME para timestamps (precisión de microsegundos)
"""