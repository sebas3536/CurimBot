# app/services/handlers/role_change/validate_role.py

from fastapi import HTTPException
from app.services.handlers.base import RoleChangeHandler
from app.services.handlers.role_change.context import RoleChangeContext

class ValidateRoleHandler(RoleChangeHandler):
    """
    Valida que el nuevo rol sea uno de los permitidos.
    """
    def _handle(self, context: RoleChangeContext):
        valid_roles = ["admin", "user"]
        if context.new_role not in valid_roles:
            raise HTTPException(
                status_code=400,
                detail=f"Rol inválido. Debe ser uno de {valid_roles}."
            )
