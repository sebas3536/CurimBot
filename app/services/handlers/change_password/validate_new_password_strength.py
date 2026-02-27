# app/services/handlers/change_password/validate_new_password_strength.py

from fastapi import HTTPException

from app.services.handlers.change_password.context import ChangePasswordContext
from app.services.handlers.base import ChangePasswordHandler

class ValidateNewPasswordStrengthHandler(ChangePasswordHandler):
    """
    Valida que la nueva contraseña cumpla con los requisitos mínimos de seguridad.
    """

    def _handle(self, context: ChangePasswordContext):
        # Ejemplo básico: verificar que la contraseña tenga al menos 8 caracteres
        if len(context.new_password) < 8:
            raise HTTPException(
                status_code=400,
                detail="La nueva contraseña debe tener al menos 8 caracteres"
            )
