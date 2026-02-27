from enum import Enum

class UserRole(str, Enum):
    admin = "admin"
    user = "user"

class LogAction(str, Enum):
    upload = "upload"
    download = "download"
    delete = "delete"
    search = "search"
    login = "login"
    ENABLE_2FA = "enable_2fa"

class FileType(str, Enum):
    pdf = "pdf"
    docx = "docx"
    txt = "txt"