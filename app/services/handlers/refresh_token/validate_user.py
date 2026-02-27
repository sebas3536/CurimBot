# app/services/handlers/refresh_token/validate_user.py

from fastapi import HTTPException
from app.models import models

from app.services.handlers.base import RefreshTokenHandler
from app.services.handlers.refresh_token.context import RefreshTokenContext

class ValidateUserHandler(RefreshTokenHandler):
    """
    Verifica que el usuario incluido en el payload del token exista en la base de datos.
    Si existe, se guarda en el contexto. Si no, lanza excepción.
    """

    def _handle(self, context: RefreshTokenContext):
        # Extrae el user_id del payload del token
        user_id = context.payload.get("sub")

        if not user_id:
            raise HTTPException(status_code=401, detail="Token inválido")

        # Busca el usuario en la base de datos
        user = context.db.query(models.User).filter(models.User.id == user_id).first()

        if not user:
            raise HTTPException(status_code=404, detail="Usuario no encontrado")

        # Asigna el usuario al contexto
        context.user = user
