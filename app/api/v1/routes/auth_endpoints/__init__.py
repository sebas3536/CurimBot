from fastapi import APIRouter
from .monitoreo import router as monitoreo_router
from .verficacion2p import router as verficacion2p_router
from .administracion import router as administracion_router
from .gestionusurios import router as gestionusurios_router
from .autentication import router as autentication_router
from .user_preferences import router as userPreferences_router
from .session_routes import router as session_router
from .password_reset_router import router as password_router



router = APIRouter()
router.include_router(autentication_router)
router.include_router(verficacion2p_router)
router.include_router(administracion_router)
router.include_router(gestionusurios_router)
router.include_router(monitoreo_router)
router.include_router(userPreferences_router)
router.include_router(session_router)
router.include_router(password_router)
