# app/services/handlers/role_change/validate_permission.py

from fastapi import HTTPException
from app.services.handlers.base import RoleChangeHandler
from app.services.handlers.role_change.context import RoleChangeContext


class ValidatePermissionHandler(RoleChangeHandler):
    """
    Valida que el usuario no intente quitarse el rol de admin a sí mismo.
    """
    def _handle(self, context: RoleChangeContext):
        # Verifica si el usuario actual está intentando quitarse su propio rol de admin
        if (
            context.current_user.id == context.target_user.id and 
            context.new_role != "admin"
        ):
            raise HTTPException(
                status_code=403, 
                detail="No puedes quitarte el rol de admin a ti mismo"
            )
