from app.services import security_service
from fastapi import HTTPException
from app.services.handlers.base import SignupContext, SignupHandler


class GenerateTokensHandler(SignupHandler):
    def _handle(self, context: SignupContext):
        user = getattr(context, "new_user", None)
        if not user:
            raise HTTPException(
                status_code=500, 
                detail="Usuario no encontrado en contexto para generación de tokens"
            )

        # Asegurarse de que user.role está cargado
        if not user.role:
            raise HTTPException(
                status_code=500,
                detail="Rol de usuario no cargado correctamente"
            )

        context.access_token = security_service.create_access_token({
            "sub": str(user.id),
            "role": user.role.name,  # Usar .name en lugar de solo .role
            "name": user.name
        })
        context.refresh_token = security_service.create_refresh_token({
            "sub": str(user.id),
            "role": user.role.name  # Usar .name en lugar de solo .role
        })