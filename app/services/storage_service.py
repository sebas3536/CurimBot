import os
from cryptography.fernet import Fernet
import logging

logger = logging.getLogger(__name__)
KEY_PATH = os.getenv("ENC_KEY_PATH", "./enc.key")

def _load_or_create_key():
    if os.path.exists(KEY_PATH):
        with open(KEY_PATH, "rb") as f:
            return f.read()
    key = Fernet.generate_key()
    with open(KEY_PATH, "wb") as f:
        f.write(key)
    return key

_FERNET = Fernet(_load_or_create_key())

def encrypt_bytes(data: bytes) -> bytes:
    return _FERNET.encrypt(data)

def decrypt_bytes(token: bytes) -> bytes:
    return _FERNET.decrypt(token)


def delete_file(file_path: str) -> bool:
    """
    Elimina un archivo físico del almacenamiento
    
    Args:
        file_path: Ruta completa del archivo a eliminar
        
    Returns:
        True si se eliminó correctamente, False si no
    """
    try:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Archivo eliminado: {file_path}")
            return True
        else:
            logger.warning(f"Archivo no encontrado para eliminar: {file_path}")
            return False
    except Exception as e:
        logger.error(f"Error eliminando archivo {file_path}: {e}")
        return False