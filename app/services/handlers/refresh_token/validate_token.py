# app/services/handlers/refresh_token/validate_token.py

from fastapi import HTTPException
from jose import jwt, JWTError

from app.services import security_service
from app.services.handlers.base import RefreshTokenHandler
from app.services.handlers.refresh_token.context import RefreshTokenContext  # Asegúrate de tener esta clase base

class ValidateRefreshTokenHandler(RefreshTokenHandler):
    """
    Valida el token de actualización (refresh token).
    Decodifica el JWT y almacena el payload en el contexto.
    """

    def _handle(self, context: RefreshTokenContext):
        try:
            # Intenta decodificar el token usando la clave secreta y algoritmo definidos
            payload = jwt.decode(
                context.refresh_token,
                security_service.SECRET_KEY,
                algorithms=[security_service.ALGORITHM]
            )
            context.payload = payload  # Guardamos el payload decodificado en el contexto

        except JWTError:
            # Si hay un error en la decodificación, lanzamos una excepción HTTP
            raise HTTPException(status_code=401, detail="Token inválido")
