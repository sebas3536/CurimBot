# app/services/handlers/role_change/context.py
class RoleChangeContext:
    """
    Contexto para el proceso de cambio de rol.
    Incluye:
    - Usuario actual (quien hace la petición)
    - Usuario objetivo (a quien se le cambia el rol)
    - El nuevo rol
    - Sesión de base de datos
    """
    
    def __init__(self, db, current_user, target_user_id, new_role):
        self.db = db
        self.current_user = current_user
        self.target_user_id = target_user_id
        self.new_role = new_role
        self.target_user = None  # Se llenará al validar existencia
