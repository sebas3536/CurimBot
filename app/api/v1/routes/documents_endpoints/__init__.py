from fastapi import APIRouter
from .upload import router as upload_router
from .download import router as download_router
from .metadata import router as metadata_router
from .dashboard import router as dashboard_router
from .search import router as search_router
from .delete import router as delete_router
from .documents import router as document_router

router = APIRouter()
router.include_router(upload_router)
router.include_router(download_router)
router.include_router(metadata_router)
router.include_router(dashboard_router)
router.include_router(search_router)
router.include_router(delete_router)
router.include_router(document_router)