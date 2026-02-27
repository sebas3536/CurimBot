# app/services/handlers/role_change/update_role.py

from app.services.handlers.base import RoleChangeHandler
from app.services.handlers.role_change.context import RoleChangeContext

class UpdateRoleHandler(RoleChangeHandler):
    """
    Actualiza el rol del usuario objetivo en la base de datos.
    """

    def _handle(self, context: RoleChangeContext):
        # Actualiza el rol
        context.target_user.role = context.new_role

        # Guarda los cambios en la base de datos
        context.db.commit()

        # Refresca el objeto en la sesión para reflejar los cambios
        context.db.refresh(context.target_user)
