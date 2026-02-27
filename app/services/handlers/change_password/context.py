# app/services/handlers/change_password/context.py
class ChangePasswordContext:
    def __init__(self, db, current_user, old_password, new_password):
        self.db = db
        self.current_user = current_user
        self.old_password = old_password
        self.new_password = new_password
