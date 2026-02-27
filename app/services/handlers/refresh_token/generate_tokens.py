# app/services/handlers/refresh_token/generate_tokens.py
from logging import Handler
from app.services import security_service
from app.services.handlers.refresh_token.context import RefreshTokenContext

class GenerateTokensHandler(Handler):
    def _handle(self, context: RefreshTokenContext):
        context.new_access_token = security_service.create_access_token({
            "sub": str(context.user.id),
            "role": context.user.role
        })
        context.new_refresh_token = security_service.create_refresh_token({
            "sub": str(context.user.id),
            "role": context.user.role
        })
