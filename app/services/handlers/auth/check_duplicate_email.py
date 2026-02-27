from fastapi import HTTPException
from app.models import models
from app.services.handlers.base import SignupContext, SignupHandler  # importa correctamente tu modelo User
from logging import Handler as LoggingHandler

class CheckDuplicateEmailHandler(SignupHandler):
    def _handle(self, context: SignupContext):
        user_exists = context.db.query(models.User).filter(models.User.email == context.user_data.email).first()
        if user_exists:
            raise HTTPException(status_code=409, detail="El correo ya está registrado")
        # No es necesario llamar al siguiente aquí, lo hace la clase base handle()
