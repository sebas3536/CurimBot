# app/services/handlers/change_password/validate_old_password.py

from fastapi import HTTPException
from app.services import security_service

from app.services.handlers.change_password.context import ChangePasswordContext
from app.services.handlers.base import ChangePasswordHandler

class ValidateOldPasswordHandler(ChangePasswordHandler):
    """
    Verifica que la contraseña actual proporcionada coincida con el hash almacenado.
    """

    def _handle(self, context: ChangePasswordContext):
        if not security_service.verify_password(context.old_password, context.current_user.password_hash):
            raise HTTPException(
                status_code=400,
                detail="Contraseña actual incorrecta"
            )
