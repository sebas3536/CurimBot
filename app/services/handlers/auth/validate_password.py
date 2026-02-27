# app/services/handlers/auth/validate_password.py

from fastapi import HTTPException
from app.services.handlers.base import SignupContext, SignupHandler  # importa correctamente tu modelo User
import re

class ValidatePasswordHandler(SignupHandler):
    def _handle(self, context: SignupContext):
        password = context.user_data.password
        password_confirm = context.user_data.password_confirm

        if password != password_confirm:
            raise HTTPException(status_code=400, detail="Las contraseñas no coinciden")

        # Validación básica de complejidad (mínimo 8 caracteres, al menos un número y una mayúscula)
        if len(password) < 8:
            raise HTTPException(status_code=400, detail="La contraseña debe tener al menos 8 caracteres")

        if not re.search(r"\d", password):
            raise HTTPException(status_code=400, detail="La contraseña debe contener al menos un número")

        if not re.search(r"[A-Z]", password):
            raise HTTPException(status_code=400, detail="La contraseña debe contener al menos una letra mayúscula")
