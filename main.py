"""
Punto de entrada principal de la aplicación FastAPI.

Gestiona:
- Ciclo de vida de la aplicación (lifespan)
- Inicialización de base de datos MySQL
- Inicialización de servicios de voz (VoiceRAGEngine)
- Inicialización de SessionStore (Redis o in-memory)
- Middleware CORS
- Montaje de routers y archivos estáticos

NOTA: recreate_database() está activado para desarrollo.
      En producción, comentar esa llamada y usar migraciones Alembic.
"""

import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text

from app.api.v1.routes.connection_manager import get_connection_manager
from app.api.v1.routes.session_store import get_session_store, init_session_store
from app.api.v1.routes.assistant import router as assist_router
from app.api.v1.routes.auth_endpoints import router as auth_router
from app.api.v1.routes.documents_endpoints import router as docs_router
from app.api.v1.routes.voice_endpoints import router as voice_router        
from app.services.Curim.voice_rag_service import get_rag_engine              # ← CAMBIADO
from app.core.config import settings
from app.core.init_roles import init_roles
from app.db.database import Base, SessionLocal, engine, init_db


load_dotenv()

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("curim")

# ─────────────────────────────────────────────
# Validación de configuración temprana
# ─────────────────────────────────────────────
try:
    settings.validate_Curim()
except ValueError as e:
    logger.error(f"Configuración inválida: {e}")
    print("\n⚠️  Curim no está completamente configurado.")
    print("📝 Por favor, agrega tu GEMINI_API_KEY en el archivo .env\n")

# ─────────────────────────────────────────────
# CORS
# ─────────────────────────────────────────────
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "http://localhost:4200").split(",")
]

# ─────────────────────────────────────────────
# Base de datos
# ─────────────────────────────────────────────

def _drop_all_tables() -> None:
    """Elimina todas las tablas respetando FK constraints."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SET FOREIGN_KEY_CHECKS = 0"))
            conn.commit()

        Base.metadata.drop_all(bind=engine)
        logger.info("Todas las tablas eliminadas")

    except Exception as e:
        logger.error(f"Error al eliminar tablas: {e}")
    finally:
        try:
            with engine.connect() as conn:
                conn.execute(text("SET FOREIGN_KEY_CHECKS = 1"))
                conn.commit()
        except Exception as e:
            logger.error(f"Error reactivando FK checks: {e}")


def recreate_database() -> None:
    """
    SOLO PARA DESARROLLO: elimina y recrea la base de datos desde cero.
    Controlar con la variable de entorno RECREATE_DB=true.
    En producción usar Alembic.
    """
    _drop_all_tables()
    Base.metadata.create_all(bind=engine)
    logger.info("Tablas recreadas")

    with SessionLocal() as db:
        init_roles(db)
        db.commit()
        logger.info("Roles inicializados")


def ensure_database() -> None:
    """
    Para producción: crea tablas solo si no existen.
    No elimina datos existentes.
    """
    Base.metadata.create_all(bind=engine)
    logger.info("Schema verificado/creado")

    with SessionLocal() as db:
        init_roles(db)
        db.commit()


# ─────────────────────────────────────────────
# Lifespan
# ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Ciclo de vida de la aplicación.

    Startup:
      1. Base de datos (recrear en dev, verificar en prod)
      2. SessionStore (Redis en prod, in-memory en dev)
      3. ConnectionManager
      4. VoiceRAGEngine (Gemini)

    Shutdown:
      - Notificar clientes activos
      - Cerrar conexiones de SessionStore
    """

    # ── STARTUP ──────────────────────────────
    logger.info("Iniciando aplicación Curim...")

    # 1. Base de datos
    recreate_db = os.getenv("RECREATE_DB", "false").lower() == "true"
    if recreate_db:
        logger.warning(
            "RECREATE_DB=true — eliminando y recreando base de datos. "
            "Desactiva esto en producción."
        )
        recreate_database()
    else:
        ensure_database()

    # 2. SessionStore
    try:
        store = init_session_store()
        logger.info(f"SessionStore listo: {type(store).__name__}")
    except Exception as e:
        logger.error(f"Error inicializando SessionStore: {e}")
        logger.warning("Continuando sin SessionStore persistente")

    # 3. ConnectionManager
    cm = get_connection_manager()
    logger.info("ConnectionManager listo")

    # 4. VoiceRAGEngine
    try:
        engine_instance = get_rag_engine()                          # ← CAMBIADO
        if engine_instance and engine_instance.is_ready:
            logger.info("VoiceRAGEngine listo")
        else:
            logger.warning(
                "VoiceRAGEngine NO está listo. "
                "Verifica GEMINI_API_KEY en .env"
            )
    except Exception as e:
        logger.error(f"Error inicializando VoiceRAGEngine: {e}", exc_info=True)

    logger.info("✅ Curim iniciado correctamente")

    # ── APLICACIÓN CORRIENDO ─────────────────
    yield

    # ── SHUTDOWN ─────────────────────────────
    logger.info("Apagando Curim...")

    try:
        notified = await cm.broadcast({
            "type": "error",
            "message": "Servidor reiniciando, reconecta en unos segundos"
        })
        logger.info(f"Clientes notificados del shutdown: {notified}")
    except Exception as e:
        logger.warning(f"Error notificando clientes en shutdown: {e}")

    try:
        s = get_session_store()
        if hasattr(s, "close"):
            await s.close()
            logger.info("SessionStore cerrado")
    except Exception as e:
        logger.warning(f"Error cerrando SessionStore: {e}")

    logger.info("Curim cerrado correctamente")


# ─────────────────────────────────────────────
# Aplicación
# ─────────────────────────────────────────────

app = FastAPI(
    title="Curim — Asistente Documental",
    description=(
        "API para gestión de documentos con asistente conversacional "
        "y voz en tiempo real con Gemini Live API."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ─────────────────────────────────────────────
# Middleware
# ─────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# Routers
# ─────────────────────────────────────────────

app.include_router(auth_router)
app.include_router(docs_router)
app.include_router(assist_router)
app.include_router(voice_router)

# ─────────────────────────────────────────────
# Endpoints de sistema
# ─────────────────────────────────────────────

@app.get("/", tags=["sistema"])
def root():
    return {
        "status": "ok",
        "service": "Curim Asistente Documental",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health", tags=["sistema"])
async def health():
    """Health check para load balancers y monitoreo."""
    cm = get_connection_manager()

    db_ok = False
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        pass

    return {
        "status": "ok" if db_ok else "degraded",
        "database": "connected" if db_ok else "error",
        "active_ws_connections": cm.active_count,
    }


# ─────────────────────────────────────────────
# Archivos estáticos
# ─────────────────────────────────────────────

app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")