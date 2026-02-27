# app/services/handlers/refresh_token/context.py
class RefreshTokenContext:
    def __init__(self, refresh_token, db):
        self.refresh_token = refresh_token
        self.db = db
        self.payload = None
        self.user = None
