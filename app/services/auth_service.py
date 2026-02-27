

import base64
from io import BytesIO
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Tuple, Dict, Any, Union
from contextlib import contextmanager
from fastapi import APIRouter, Body, Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer
import pyotp
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
from jose import JWTError, jwt
from typing import Union, Tuple, Dict
from app.services import security_service

import qrcode                   
import secrets   


from app.core.security import (
    AccountLockedError,
    ActiveSessionInfo, 
    ActiveSessionsResponse,
    InvalidCredentialsError,
    PermissionDeniedError, 
    RefreshTokenRequest, 
    ResetPasswordConfirm, 
    ResetPasswordRequest, 
    RevokeSessionRequest, 
    Token,
    TokenBlacklistedError,
    TokenExpiredError, 
    TrustedDeviceRequest, 
    TwoFactorSetupRequest, 
    TwoFactorVerifyRequest,
    UserAlreadyExistsError,
    UserNotFoundError,
    WeakPasswordError,
)

from app.db.database import get_db
from app.models.models import Log, Role , User, BlacklistedToken, LoginAttempt
from app.enums.enums import LogAction, UserRole
from app.schemas.user_schemas import UserCreate
from app.services import security_service
from app.services.session_service import SessionService

# ========================================
# 🔧 CONFIGURACIÓN INICIAL
# ========================================

# Esquema OAuth2
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

# Logger
logger = logging.getLogger(__name__)

# ========================================
#  SERVICIO DE AUTENTICACIÓN
# ========================================

class AuthService:
    """
    Servicio completo de autenticación y gestión de usuarios
    
    Proporciona funcionalidades para:
    - Registro y login de usuarios
    - Gestión de tokens y sesiones
    - Control de seguridad y auditoría
    - Administración de usuarios
    """
    
    # Configuración de seguridad
    MAX_LOGIN_ATTEMPTS = 5
    LOCKOUT_DURATION = timedelta(minutes=15)
    TOKEN_BLACKLIST_CLEANUP_HOURS = 24
    PASSWORD_MIN_LENGTH = 8
    
    @staticmethod
    @contextmanager
    def db_transaction(db: Session):
        """
        Context manager para transacciones de base de datos con rollback automático
        
        Args:
            db: Sesión de base de datos
            
        Yields:
            Sesión de base de datos
        """
        try:
            yield db
            db.commit()
        except Exception as e:
            db.rollback()
            logger.error(f"Database transaction failed: {e}")
            raise

    @staticmethod
    def _validate_password_strength(password: str) -> None:
        """
        Valida que la contraseña cumpla con los requisitos de seguridad
        
        Args:
            password: Contraseña a validar
            
        Raises:
            WeakPasswordError: Si la contraseña no cumple los requisitos
        """
        if len(password) < AuthService.PASSWORD_MIN_LENGTH:
            raise WeakPasswordError(
                f"La contraseña debe tener al menos {AuthService.PASSWORD_MIN_LENGTH} caracteres"
            )
        
        if not any(c.isupper() for c in password):
            raise WeakPasswordError("La contraseña debe contener al menos una letra mayúscula")
        
        if not any(c.islower() for c in password):
            raise WeakPasswordError("La contraseña debe contener al menos una letra minúscula")
        
        if not any(c.isdigit() for c in password):
            raise WeakPasswordError("La contraseña debe contener al menos un número")
        
        special_chars = "!@#$%^&*()_+-=[]{}|;:,.<>?"
        if not any(c in special_chars for c in password):
            raise WeakPasswordError("La contraseña debe contener al menos un carácter especial")

    @staticmethod
    def _check_account_lockout(email: str, db: Session) -> None:
        """
        Verifica si la cuenta está bloqueada por demasiados intentos fallidos
        
        Args:
            email: Email del usuario
            db: Sesión de base de datos
            
        Raises:
            AccountLockedError: Si la cuenta está bloqueada
        """
        cutoff_time = datetime.utcnow() - AuthService.LOCKOUT_DURATION
        
        failed_attempts = (
            db.query(LoginAttempt)
            .filter(
                LoginAttempt.email == email,
                LoginAttempt.success == False,
                LoginAttempt.attempted_at > cutoff_time
            )
            .count()
        )
        
        if failed_attempts >= AuthService.MAX_LOGIN_ATTEMPTS:
            remaining_lockout = AuthService.LOCKOUT_DURATION - (datetime.utcnow() - cutoff_time)
            raise AccountLockedError(
                f"Cuenta bloqueada por demasiados intentos fallidos. "
                f"Intente nuevamente en {remaining_lockout.seconds // 60} minutos."
            )

    @staticmethod
    def _log_login_attempt(
        email: str, 
        success: bool, 
        ip_address: str = None, 
        user_agent: str = None, 
        db: Session = None
    ) -> None:
        """
        Registra intento de login para auditoría de seguridad
        
        Args:
            email: Email del usuario
            success: Si el intento fue exitoso
            ip_address: Dirección IP del cliente
            user_agent: User-Agent del cliente
            db: Sesión de base de datos
        """
        try:
            if db:
                attempt = LoginAttempt(
                    email=email,
                    success=success,
                    ip_address=ip_address,
                    user_agent=user_agent,
                    attempted_at=datetime.utcnow()
                )
                db.add(attempt)
                db.commit()
        except Exception as e:
            logger.warning(f"Failed to log login attempt: {e}")

    @staticmethod
    def _cleanup_expired_tokens(db: Session) -> None:
        """
        Limpia tokens en lista negra expirados
        
        Args:
            db: Sesión de base de datos
        """
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=AuthService.TOKEN_BLACKLIST_CLEANUP_HOURS)
            expired_tokens = db.query(BlacklistedToken).filter(
                BlacklistedToken.blacklisted_at < cutoff_time
            )
            count = expired_tokens.count()
            if count > 0:
                expired_tokens.delete()
                db.commit()
        except Exception as e:
            logger.warning(f"Failed to cleanup expired tokens: {e}")

    @staticmethod
    def decode_and_validate_token(token: str, db: Session) -> dict:
        """
        Decodifica el token JWT y valida su validez.
        
        Args:
            token (str): Token JWT a decodificar.
            db (Session): Sesión de base de datos para verificar blacklist o usuario.
            
        Returns:
            dict: Payload decodificado del token.
        
        Raises:
            TokenExpiredError: Si el token está expirado.
            TokenBlacklistedError: Si el token está en la blacklist.
            HTTPException: Para errores generales.
        """
        try:
            # Decodifica token usando tu servicio de seguridad
            payload = security_service.decode_token(token)
            
            # Validar blacklist si tienes esa funcionalidad
            blacklisted = db.query(BlacklistedToken).filter(
                BlacklistedToken.token == token
            ).first()
            if blacklisted:
                raise TokenBlacklistedError("Token está en blacklist")
            
            # Validar que el token tenga "sub" (user_id)
            if "sub" not in payload:
                raise TokenExpiredError("Token inválido")
            
            return payload
        
        except JWTError as e:
            # JWTError viene de PyJWT o similar
            raise TokenExpiredError("Token inválido o expirado")
        
        except Exception as e:
            logger.exception(f"Error decoding token: {e}")
            raise HTTPException(status_code=500, detail="Error interno del servidor")

    
    @staticmethod
    def logout_user(refresh_token: str, db: Session) -> bool:
        """
        Añade el refresh token a la lista negra para invalidarlo.

        Args:
            refresh_token: El token de refresco a invalidar
            db: Sesión de base de datos

        Returns:
            True si el token fue añadido correctamente a la blacklist.
        """
        try:
            blacklisted_token = BlacklistedToken(
                token=refresh_token,
                blacklisted_at=datetime.utcnow()
            )
            db.add(blacklisted_token)
            db.commit()
            return True
        except Exception as e:
            logger.error(f"Error during logout (blacklisting token): {e}")
            db.rollback()
            return False
        
    @staticmethod
    def _record_failed_attempt(user: User, ip_address: str, user_agent: str, db: Session) -> None:
        """
        Registra un intento fallido de login y bloquea la cuenta si es necesario
        
        Args:
            user: Usuario que intentó hacer login
            ip_address: IP del intento
            user_agent: User-Agent del intento
            db: Sesión de base de datos
        """
        try:
            # Incrementar contador de intentos fallidos
            user.failed_attempts += 1
            
            # Si alcanza el límite, bloquear cuenta
            if user.failed_attempts >= AuthService.MAX_LOGIN_ATTEMPTS:
                user.locked_until = datetime.utcnow() + AuthService.LOCKOUT_DURATION
                logger.warning(
                    f"Account locked for user {user.email} until {user.locked_until} "
                    f"due to {user.failed_attempts} failed attempts"
                )
            
            # Registrar en tabla de intentos de login
            AuthService._log_login_attempt(
                email=user.email,
                success=False,
                ip_address=ip_address,
                user_agent=user_agent,
                db=db
            )
            
            db.commit()
            
        except Exception as e:
            logger.error(f"Error recording failed attempt for {user.email}: {e}")
            db.rollback()

    @staticmethod
    def login_user(email: str, password: str, db: Session, ip_address: str, user_agent: str):
        """Autenticar usuario y generar tokens"""
        
        # 1. Buscar usuario
        user = db.query(User).filter(User.email == email).first()
        if not user:
            raise InvalidCredentialsError("Credenciales inválidas")
        
        # 2. ✅ Verificar si la cuenta está bloqueada temporalmente
        if user.locked_until and user.locked_until > datetime.utcnow():
            remaining_time = user.locked_until - datetime.utcnow()
            minutes_remaining = int(remaining_time.total_seconds() / 60)
            raise AccountLockedError(
                f"Cuenta bloqueada temporalmente. Intente nuevamente en {minutes_remaining} minutos."
            )
        
        # 3. Si ya pasó el tiempo de bloqueo, resetear contador
        if user.locked_until and user.locked_until <= datetime.utcnow():
            user.failed_attempts = 0
            user.locked_until = None
            db.commit()
        
        # 4. ✅ Verificar contraseña (usando security_service)
        if not security_service.verify_password(password, user.password_hash):
            # Registrar intento fallido
            AuthService._record_failed_attempt(user, ip_address, user_agent, db)
            raise InvalidCredentialsError("Credenciales inválidas")
        
        # 5. Verificar cuenta activa
        if not user.is_active:
            raise AccountLockedError("Cuenta desactivada")
        
        # 6. Verificar 2FA
        if user.two_factor_enabled:
            return {
                "requires_2fa": True,
                "message": "Verificación de dos factores requerida"
            }
        
        # 7. ✅ Resetear intentos fallidos en login exitoso
        user.failed_attempts = 0
        user.locked_until = None
        
        # 8. Generar tokens
        token_data = {
            "sub": str(user.id),
            "role": user.role.name if hasattr(user.role, 'name') else str(user.role),
            "email": user.email
        }
        
        access_token = security_service.create_access_token(token_data)
        refresh_token = security_service.create_refresh_token(token_data)
        
        # 9. Actualizar último login
        user.last_login = datetime.utcnow()
        db.commit()
        
        # 10. Registrar intento exitoso
        AuthService._log_login_attempt(
            email=user.email,
            success=True,
            ip_address=ip_address,
            user_agent=user_agent,
            db=db
        )
        
        return access_token, refresh_token

    @staticmethod
    def update_last_login(user: User, db: Session) -> None:
        """
        Actualiza la fecha y hora del último login del usuario.
        
        Args:
            user: instancia del usuario
            db: sesión de base de datos
        """
        try:
            user.last_login = datetime.utcnow()
            db.add(user)
            db.commit()
        except Exception as e:
            logger.warning(f"Failed to update last login for user {user.email}: {e}")
            db.rollback()

# ========================================
#  MÉTODOS PRINCIPALES DE USUARIO
# ========================================

    @staticmethod
    def signup_user(
        user_data: UserCreate, 
        db: Session, 
        ip_address: str = None
    ) -> Tuple[str, str]:
        """
        Registra un nuevo usuario con validación completa
        
        Args:
            user_data: Datos del usuario a registrar
            db: Sesión de base de datos
            ip_address: Dirección IP del cliente
            
        Returns:
            Tuple[str, str]: Token de acceso y token de refresh
            
        Raises:
            UserAlreadyExistsError: Si el usuario ya existe
            WeakPasswordError: Si la contraseña es débil
        """
        try:
            # Validar fortaleza de contraseña
            AuthService._validate_password_strength(user_data.password)
            
            # Verificar confirmación de contraseña
            if user_data.password != user_data.password_confirm:
                raise WeakPasswordError("Las contraseñas no coinciden")

            # Verificar si el usuario ya existe
            existing_user = db.query(User).filter(User.email == user_data.email.lower()).first()
            if existing_user:
                raise UserAlreadyExistsError("Ya existe un usuario con este email")

            with AuthService.db_transaction(db):
                # Verificar si es el primer usuario
                admin_role = db.query(Role).filter(Role.name == "admin").first()
                user_role = db.query(Role).filter(Role.name == "user").first()
                if not admin_role:
                    admin_role = Role(name="admin", description="Administrator role")
                    db.add(admin_role)
                    db.flush()
                if not user_role:
                    user_role = Role(name="user", description="Regular user role")
                    db.add(user_role)
                    db.flush()
                is_first_user = db.query(User).count() == 0
                
                # Crear usuario
                hashed_password = security_service.hash_password(user_data.password)
                new_user = User(
                    name=user_data.name.strip(),
                    email=user_data.email.lower().strip(),
                    password_hash=hashed_password,
                    role_id=admin_role.id if is_first_user else user_role.id,  # Usar role_id
                    created_at=datetime.utcnow(),
                    is_active=True
                )
                

                db.add(new_user)
                db.flush()  # Obtener ID de usuario sin commit
                
                # Generar tokens
                access_token = security_service.create_access_token({
                    "sub": str(new_user.id),
                    "role": new_user.role.name,
                    "email": new_user.email
                })
                refresh_token = security_service.create_refresh_token({
                    "sub": str(new_user.id),
                    "role": new_user.role.name
                })
                
                # Registrar registro exitoso
                AuthService._log_login_attempt(
                    new_user.email, True, ip_address, db=db
                )
                
                return access_token, refresh_token

        except (UserAlreadyExistsError, WeakPasswordError):
            raise
        except IntegrityError as e:
            logger.error(f"Database integrity error during signup: {e}")
            raise UserAlreadyExistsError("Error al crear el usuario - email ya existe")
        except SQLAlchemyError as e:
            logger.error(f"Database error during signup: {e}")
            raise HTTPException(
                status_code=500,
                detail="Error interno del servidor durante el registro"
            )
        except Exception as e:
            logger.exception(f"Unexpected error during signup: {e}")
            raise HTTPException(
                status_code=500,
                detail="Error interno del servidor"
            )


    @staticmethod
    def refresh_tokens(refresh_token: str, db: Session) -> Tuple[str, str]:
        """
        Renueva tokens de acceso y refresh
        
        Args:
            refresh_token: Token de refresh actual
            db: Sesión de base de datos
            
        Returns:
            Tuple[str, str]: Nuevo token de acceso y refresh
            
        Raises:
            TokenExpiredError: Si el token está expirado
            TokenBlacklistedError: Si el token está en lista negra
        """
        try:
            # Validar token de refresh
            try:
                payload = jwt.decode(
                    refresh_token,
                    security_service.SECRET_KEY,
                    algorithms=[security_service.ALGORITHM]
                )
            except JWTError as e:
                logger.warning(f"Invalid refresh token: {e}")
                raise TokenExpiredError("Refresh token inválido o expirado")
            
            # Verificar si el token está en lista negra
            if AuthService.is_token_blacklisted(refresh_token, db):
                raise TokenBlacklistedError("Refresh token ha sido revocado")
            
            # Obtener usuario
            user_id = payload.get("sub")
            if not user_id:
                raise TokenExpiredError("Token inválido")
            
            user = db.query(User).filter(User.id == user_id).first()
            if not user or not user.is_active:
                raise UserNotFoundError("Usuario no encontrado o desactivado")
            
            # Generar nuevos tokens
            token_data = {
                "sub": str(user.id),
                "role": user.role.value,
                "email": user.email
            }
            
            new_access_token = security_service.create_access_token(token_data)
            new_refresh_token = security_service.create_refresh_token(token_data)
            
            # Agregar token viejo a lista negra
            with AuthService.db_transaction(db):
                blacklisted_token = BlacklistedToken(
                    token=refresh_token,
                    blacklisted_at=datetime.utcnow()
                )
                db.add(blacklisted_token)
            
            return new_access_token, new_refresh_token

        except (TokenExpiredError, TokenBlacklistedError, UserNotFoundError):
            raise
        except SQLAlchemyError as e:
            logger.error(f"Database error during token refresh: {e}")
            raise HTTPException(
                status_code=500,
                detail="Error interno del servidor durante la renovación de tokens"
            )
        except Exception as e:
            logger.exception(f"Unexpected error during token refresh: {e}")
            raise HTTPException(
                status_code=500,
                detail="Error interno del servidor"
            )

# ========================================
#  MÉTODOS DE SEGURIDAD
# ========================================

    @staticmethod
    def change_password(
        user: User, 
        old_password: str, 
        new_password: str, 
        db: Session
    ) -> None:
        """
        Cambia la contraseña del usuario con validación
        
        Args:
            user: Usuario actual
            old_password: Contraseña actual
            new_password: Nueva contraseña
            db: Sesión de base de datos
            
        Raises:
            InvalidCredentialsError: Si la contraseña actual es incorrecta
            WeakPasswordError: Si la nueva contraseña es débil
        """
        try:
            # Verificar contraseña actual
            if not security_service.verify_password(old_password, user.password_hash):
                raise InvalidCredentialsError("Contraseña actual incorrecta")
            
            # Validar fortaleza de nueva contraseña
            AuthService._validate_password_strength(new_password)
            
            # Verificar que la nueva contraseña sea diferente
            if security_service.verify_password(new_password, user.password_hash):
                raise WeakPasswordError("La nueva contraseña debe ser diferente a la actual")
            
            with AuthService.db_transaction(db):
                # Actualizar contraseña
                user.password_hash = security_service.hash_password(new_password)
                user.password_changed_at = datetime.utcnow()
                

        except (InvalidCredentialsError, WeakPasswordError):
            raise
        except SQLAlchemyError as e:
            logger.error(f"Database error during password change: {e}")
            raise HTTPException(
                status_code=500,
                detail="Error interno del servidor durante el cambio de contraseña"
            )
        except Exception as e:
            logger.exception(f"Unexpected error during password change: {e}")
            raise HTTPException(
                status_code=500,
                detail="Error interno del servidor"
            )

    @staticmethod
    def is_token_blacklisted(token: str, db: Session) -> bool:
        """
        Verifica si un token está en lista negra
        
        Args:
            token: Token a verificar
            db: Sesión de base de datos
            
        Returns:
            bool: True si el token está en lista negra
        """
        try:
            return db.query(BlacklistedToken).filter(
                BlacklistedToken.token == token
            ).first() is not None
        except SQLAlchemyError as e:
            logger.error(f"Database error checking blacklisted token: {e}")
            # En caso de error, asumir que no está en lista negra
            return False

    @staticmethod
    def decode_and_validate_token(token: str, db: Session) -> Dict[str, Any]:
        """
        Decodifica y valida un token JWT
        
        Args:
            token: Token JWT a validar
            db: Sesión de base de datos
            
        Returns:
            Dict[str, Any]: Payload del token decodificado
            
        Raises:
            TokenBlacklistedError: Si el token está en lista negra
            TokenExpiredError: Si el token es inválido o expirado
        """
        try:
            # Verificar lista negra primero
            if AuthService.is_token_blacklisted(token, db):
                raise TokenBlacklistedError("Token ha sido revocado")
            
            # Decodificar token
            payload = jwt.decode(
                token,
                security_service.SECRET_KEY,
                algorithms=[security_service.ALGORITHM]
            )
            
            return payload
            
        except JWTError as e:
            logger.warning(f"Invalid token: {e}")
            raise TokenExpiredError("Token inválido o expirado")
        except TokenBlacklistedError:
            raise

    @staticmethod
    def get_user_from_token(token: str, db: Session) -> User:
        """
        Obtiene usuario desde un token válido
        
        Args:
            token: Token JWT
            db: Sesión de base de datos
            
        Returns:
            User: Usuario autenticado
            
        Raises:
            TokenExpiredError: Si el token es inválido
            UserNotFoundError: Si el usuario no existe o está inactivo
        """
        try:
            payload = AuthService.decode_and_validate_token(token, db)
            
            user_id = payload.get("sub")
            if not user_id:
                raise TokenExpiredError("Token inválido")
            
            user = db.query(User).filter(User.id == user_id).first()
            if not user:
                raise UserNotFoundError("Usuario no encontrado")
            
            if not user.is_active:
                raise UserNotFoundError("Usuario desactivado")
            
            return user

        except (TokenExpiredError, TokenBlacklistedError, UserNotFoundError):
            raise
        except Exception as e:
            logger.exception(f"Unexpected error getting user from token: {e}")
            raise HTTPException(
                status_code=500,
                detail="Error interno del servidor"
            )

    # ========================================
    # 👑 MÉTODOS DE ADMINISTRACIÓN
    # ========================================

    @staticmethod
    def update_user_role(
        admin_user: User, 
        target_user_id: int, 
        new_role_name: str,
        db: Session
    ) -> User:
        """
        Actualiza el rol de usuario con validación de permisos
        
        Args:
            admin_user: Usuario administrador que realiza la acción
            target_user_id: ID del usuario objetivo
            new_role: Nuevo rol a asignar
            db: Sesión de base de datos
            
        Returns:
            User: Usuario actualizado
            
        Raises:
            PermissionDeniedError: Si no tiene permisos de administrador
            UserNotFoundError: Si el usuario objetivo no existe
        """
        try:
            # Verificar permisos de administrador
            if admin_user.role.name != "admin":
                raise PermissionDeniedError("Se requieren permisos de administrador")
        
            # Obtener usuario objetivo
            target_user = db.query(User).filter(User.id == target_user_id).first()
            if not target_user:
                raise UserNotFoundError("Usuario objetivo no encontrado")
            
            # Prevenir auto-degradación
            if admin_user.id == target_user_id and new_role_name != "admin":
                raise PermissionDeniedError("No puede modificar su propio rol de administrador")
            
            # Obtener el nuevo rol de la base de datos
            new_role = db.query(Role).filter(Role.name == new_role_name).first()
            if not new_role:
                raise HTTPException(
                    status_code=400,
                    detail="Rol inválido"
                )
            
            with AuthService.db_transaction(db):
                old_role_name = target_user.role.name
                target_user.role_id = new_role.id  # Actualizar role_id en lugar de role
                # Si tienes estos campos en tu modelo:
                # target_user.role_changed_at = datetime.utcnow()
                # target_user.role_changed_by = admin_user.id
                
            
            db.refresh(target_user)  # Refrescar para obtener la relación actualizada
            return target_user

        except (PermissionDeniedError, UserNotFoundError):
            raise
        except SQLAlchemyError as e:
            logger.error(f"Database error during role update: {e}")
            raise HTTPException(
                status_code=500,
                detail="Error interno del servidor durante la actualización de rol"
            )

    @staticmethod
    def deactivate_user(admin_user: User, target_user_id: int, db: Session) -> User:
        """
        Desactiva cuenta de usuario
        
        Args:
            admin_user: Usuario administrador que realiza la acción
            target_user_id: ID del usuario a desactivar
            db: Sesión de base de datos
            
        Returns:
            User: Usuario desactivado
            
        Raises:
            PermissionDeniedError: Si no tiene permisos de administrador
            UserNotFoundError: Si el usuario no existe
        """
        try:
            if admin_user.role.name != "admin":  # Cambiar .value por .name
                raise PermissionDeniedError("Se requieren permisos de administrador")
            
            target_user = db.query(User).filter(User.id == target_user_id).first()
            if not target_user:
                raise UserNotFoundError("Usuario no encontrado")
            
            if admin_user.id == target_user_id:
                raise PermissionDeniedError("No puede desactivar su propia cuenta")
            
            with AuthService.db_transaction(db):
                target_user.is_active = False
                target_user.deactivated_at = datetime.utcnow()
                target_user.deactivated_by = admin_user.id
            
            return target_user

        except (PermissionDeniedError, UserNotFoundError):
            raise
        except Exception as e:
            logger.exception(f"Error deactivating user: {e}")
            raise HTTPException(status_code=500, detail="Error interno del servidor")

# ========================================
# MÉTODOS DE MONITOREO Y AUDITORÍA
# ========================================

    @staticmethod
    def get_login_attempts_stats(
        email: str, 
        db: Session, 
        hours: int = 24
    ) -> Dict[str, Any]:
        """
        Obtiene estadísticas de intentos de login para monitoreo
        
        Args:
            email: Email del usuario
            db: Sesión de base de datos
            hours: Período de tiempo en horas
            
        Returns:
            Dict[str, Any]: Estadísticas de intentos de login
        """
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            
            attempts = db.query(LoginAttempt).filter(
                LoginAttempt.email == email,
                LoginAttempt.attempted_at > cutoff_time
            ).all()
            
            total_attempts = len(attempts)
            successful_attempts = len([a for a in attempts if a.success])
            failed_attempts = total_attempts - successful_attempts
            
            return {
                "email": email,
                "period_hours": hours,
                "total_attempts": total_attempts,
                "successful_attempts": successful_attempts,
                "failed_attempts": failed_attempts,
                "success_rate": (successful_attempts / total_attempts * 100) if total_attempts > 0 else 0,
                "last_attempt": max([a.attempted_at for a in attempts]) if attempts else None
            }
            
        except Exception as e:
            logger.error(f"Error getting login stats for {email}: {e}")
            return {"error": "Could not retrieve stats"}


# ========================================
# DEPENDENCIAS DE SEGURIDAD
# ========================================

def get_current_user(
        token: str = Depends(oauth2_scheme),
        db: Session = Depends(get_db)
    ) -> User:
        """
        Dependencia para obtener usuario autenticado desde token
        
        Args:
            token: Token JWT
            db: Sesión de base de datos
            
        Returns:
            User: Usuario autenticado
            
        Raises:
            HTTPException: Si hay error de autenticación
        """
        try:
            user = AuthService.get_user_from_token(token, db)
            return user
        except TokenExpiredError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token expirado o inválido",
                headers={"WWW-Authenticate": "Bearer"}
            )
        except TokenBlacklistedError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token revocado",
                headers={"WWW-Authenticate": "Bearer"}
            )
        except UserNotFoundError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Usuario no encontrado o desactivado",
                headers={"WWW-Authenticate": "Bearer"}
            )
        except Exception as e:
            logger.exception(f"Unexpected error in get_current_user: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error interno del servidor"
            )


def require_admin(current_user: User = Depends(get_current_user)) -> User:
        """
        Dependencia que requiere rol de administrador
        
        Args:
            current_user: Usuario autenticado
            
        Returns:
            User: Usuario administrador
            
        Raises:
            HTTPException: Si no tiene permisos de administrador
        """
        if current_user.role.name != "admin":  # Cambiar .value por .name
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Se requieren permisos de administrador"
            )
        return current_user


def get_client_info(request: Request) -> tuple[str, str]:
        """
        Extrae información del cliente desde la request
        
        Args:
            request: Request de FastAPI
            
        Returns:
            tuple[str, str]: Dirección IP y User-Agent
        """
        ip_address = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")
        return ip_address, user_agent

class TwoFactorAuthService:
    """Servicio para manejar autenticación de dos factores"""

    @staticmethod
    def generate_secret() -> str:
        """Genera un secreto aleatorio para TOTP"""
        return pyotp.random_base32()

    @staticmethod
    def generate_qr_code(user_email: str, secret: str, issuer_name: str = "SecureDocApp") -> str:
        """
        Genera un código QR para configurar 2FA en Google Authenticator

        Args:
            user_email: Email del usuario
            secret: Secreto TOTP
            issuer_name: Nombre de la aplicación

        Returns:
            String base64 de la imagen QR
        """
        totp_uri = pyotp.totp.TOTP(secret).provisioning_uri(
            name=user_email,
            issuer_name=issuer_name
        )
        
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        qr.add_data(totp_uri)
        qr.make(fit=True)

        img = qr.make_image(fill_color="black", back_color="white")

        buffer = BytesIO()
        img.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode()

        return img_str

    @staticmethod
    def verify_totp_code(secret: str, code: str, window: int = 1) -> bool:
        """
        Verifica un código TOTP

        Args:
            secret: Secreto TOTP del usuario
            code: Código ingresado por el usuario
            window: Ventana de tiempo para validación (períodos de 30s)

        Returns:
            True si el código es válido
        """
        totp = pyotp.TOTP(secret)
        return totp.verify(code, valid_window=window)

    @staticmethod
    def enable_2fa_for_user(user: 'User', db: 'Session') -> Tuple[str, str]:
        """
        Habilita 2FA para un usuario

        Returns:
            Tupla (secret, qr_code_base64)
        """
        secret = TwoFactorAuthService.generate_secret()

        user.two_factor_secret_temp = secret
        user.two_factor_enabled = False  # Se activará después de verificar
        db.commit()

        qr_code = TwoFactorAuthService.generate_qr_code(user.email, secret)

        return secret, qr_code

    @staticmethod
    def confirm_2fa_setup(user: 'User', code: str, db: 'Session') -> bool:
        """
        Confirma la configuración de 2FA verificando el primer código

        Returns:
            True si la configuración fue exitosa
        """
        if not user.two_factor_secret_temp:
            raise ValueError("No hay configuración de 2FA pendiente")

        if TwoFactorAuthService.verify_totp_code(user.two_factor_secret_temp, code):
            user.two_factor_secret = user.two_factor_secret_temp
            user.two_factor_secret_temp = None
            user.two_factor_enabled = True
            user.two_factor_enabled_at = datetime.utcnow()

            # Crear log de activación de 2FA (asegúrate que Log y LogAction estén importados)
            log = Log(
                user_id=user.id,
                action=LogAction.ENABLE_2FA,
                detail="Activación de autenticación en dos pasos"
            )
            db.add(log)

            db.commit()
            return True

        return False

    @staticmethod
    def disable_2fa_for_user(user: 'User', code: str, db: 'Session') -> bool:
        """
        Deshabilita 2FA para un usuario (requiere código válido)

        Returns:
            True si se deshabilitó exitosamente
        """
        if not user.two_factor_enabled or not user.two_factor_secret:
            raise ValueError("2FA no está habilitado")

        if TwoFactorAuthService.verify_totp_code(user.two_factor_secret, code):
            user.two_factor_enabled = False
            user.two_factor_secret = None
            user.two_factor_disabled_at = datetime.utcnow()
            db.commit()
            return True

        return False

    @staticmethod
    def generate_backup_codes(count: int = 8) -> List[str]:
        """
        Genera códigos de respaldo para 2FA

        Args:
            count: Número de códigos a generar

        Returns:
            Lista de códigos de respaldo
        """
        import secrets
        return [secrets.token_hex(4).upper() for _ in range(count)]

    @staticmethod
    def save_backup_codes(user: 'User', codes: List[str], db: 'Session'):
        """
        Guarda códigos de respaldo hasheados
        """
        from app.core.security import get_password_hash

        hashed_codes = [get_password_hash(code) for code in codes]
        user.backup_codes = ",".join(hashed_codes)
        db.commit()

    @staticmethod
    def verify_backup_code(user: 'User', code: str, db: 'Session') -> bool:
        """
        Verifica y consume un código de respaldo

        Returns:
            True si el código es válido
        """
        from app.core.security import verify_password

        if not user.backup_codes:
            return False

        codes = user.backup_codes.split(",")

        for i, hashed_code in enumerate(codes):
            if verify_password(code, hashed_code):
                codes.pop(i)
                user.backup_codes = ",".join(codes) if codes else None
                db.commit()
                return True

        return False
    
class TwoFactorRequiredError(Exception):
    """Excepción lanzada cuando se requiere verificación de dos factores"""
    pass