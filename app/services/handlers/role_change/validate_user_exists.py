# app/services/handlers/role_change/validate_user_exists.py

from fastapi import HTTPException
from app.models import models

from app.services.handlers.base import RoleChangeHandler
from app.services.handlers.role_change.context import RoleChangeContext

class ValidateUserExistsHandler(RoleChangeHandler):
    """
    Verifica que el usuario objetivo exista en la base de datos.
    Si existe, lo asigna al contexto; si no, lanza excepción 404.
    """
    def _handle(self, context: RoleChangeContext):
        user = context.db.query(models.User).filter(models.User.id == context.target_user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="Usuario no encontrado")
        context.target_user = user
