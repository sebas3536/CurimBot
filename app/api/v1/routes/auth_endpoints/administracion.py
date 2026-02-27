"""
Módulo de autenticación y gestión de usuarios.

Este módulo contiene los endpoints relacionados con la autenticación de usuarios,
gestión de roles, activación/desactivación de cuentas y estadísticas de inicio de sesión.
Todos los endpoints administrativos requieren privilegios de administrador.
"""

from app.services.security_service import verify_password
import logging
from datetime import datetime
from typing import List, Optional
from .....schemas.auth_schemas import (
    ActiveSessionsResponse, 
    BackupCodesResponse, 
    RefreshTokenRequest, 
    ResetPasswordRequest, 
    Token, 
    TwoFactorConfirmRequest, 
    TwoFactorDisableRequest, 
    TwoFactorSetupResponse, 
    TwoFactorVerifyRequest
)

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status, Body
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.models.models import User
from app.enums.enums import UserRole

from app.schemas.common_schemas import LoginStatsResponse
from app.schemas.user_schemas import UserCreate, UserInfoResponse, UserManagementResponse
from app.services.auth_service import (
    AccountLockedError, 
    AuthService, 
    InvalidCredentialsError, 
    PermissionDeniedError, 
    TokenBlacklistedError, 
    TokenExpiredError, 
    TwoFactorAuthService, 
    UserAlreadyExistsError, 
    UserNotFoundError, 
    WeakPasswordError, 
    get_client_info, 
    get_current_user, 
    require_admin
)


# ========================================
# CONFIGURACIÓN
# ========================================

router = APIRouter(prefix="/auth", tags=["authentication"])
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")
logger = logging.getLogger(__name__)


# ========================================
# ENDPOINTS DE ADMINISTRACIÓN
# ========================================

@router.get("/users", response_model=List[UserInfoResponse])
def get_all_users(
    skip: int = 0,
    limit: int = 100,
    active_only: bool = True,
    admin_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """
    Obtener lista de usuarios del sistema.
    
    Este endpoint permite a los administradores consultar todos los usuarios registrados
    en el sistema con soporte para paginación y filtrado por estado de activación.
    
    Args:
        skip (int): Número de usuarios a omitir para paginación. Debe ser >= 0. Por defecto 0.
        limit (int): Número máximo de usuarios a retornar. Rango válido: 1-100. Por defecto 100.
        active_only (bool): Si es True, solo retorna usuarios activos. Por defecto True.
        admin_user (User): Usuario administrador autenticado (inyectado automáticamente).
        db (Session): Sesión de base de datos (inyectada automáticamente).
    
    Returns:
        List[UserInfoResponse]: Lista de usuarios con su información básica incluyendo:
            - ID del usuario
            - Email
            - Nombre
            - Rol
            - Fecha de creación
            - Último inicio de sesión
            - Estado de activación
            - Estado de autenticación de dos factores
    
    Raises:
        HTTPException 400: Si los parámetros de paginación son inválidos
        HTTPException 403: Si el usuario no tiene privilegios de administrador
        HTTPException 500: Si ocurre un error interno del servidor
    
    Example:
        GET /auth/users?skip=0&limit=50&active_only=true
    """
    try:
        # Validar parámetros de paginación
        if skip < 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Skip no puede ser negativo"
            )
        if limit <= 0 or limit > 100:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Limit debe estar entre 1 y 100"
            )
        
        # Construir consulta base
        query = db.query(User)
        if active_only:
            query = query.filter(User.is_active == True)
            
        # Ejecutar consulta con paginación
        users = query.offset(skip).limit(limit).all()
        
        # Transformar resultados a modelo de respuesta
        return [
            UserInfoResponse(
                id=user.id,
                email=user.email,
                name=user.name,
                role=user.role.name,
                created_at=user.created_at,
                last_login=user.last_login,
                is_active=user.is_active,
                two_factor_enabled=user.two_factor_enabled
            )
            for user in users
        ]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error getting all users: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error interno del servidor"
        )


@router.patch("/users/{user_id}/role", response_model=UserManagementResponse)
def update_user_role(
    user_id: int,
    new_role: str = Body(..., embed=True, description="New role for the user (admin or user)"),
    admin_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """
    Actualizar el rol de un usuario.
    
    Este endpoint permite a los administradores cambiar el rol de cualquier usuario
    en el sistema. Los roles válidos son 'admin' y 'user'.
    
    Args:
        user_id (int): ID del usuario cuyo rol se desea actualizar. Debe ser > 0.
        new_role (str): Nuevo rol a asignar. Valores válidos: 'admin' o 'user'.
        admin_user (User): Usuario administrador autenticado (inyectado automáticamente).
        db (Session): Sesión de base de datos (inyectada automáticamente).
    
    Returns:
        UserManagementResponse: Información de la operación incluyendo:
            - Mensaje de confirmación
            - ID del usuario modificado
            - Email del usuario modificado
            - Nuevo rol asignado
            - Email del administrador que realizó la operación
            - Timestamp de la actualización
    
    Raises:
        HTTPException 400: Si el user_id es inválido o el rol no es válido
        HTTPException 403: Si el usuario no tiene permisos o intenta modificar su propio rol
        HTTPException 404: Si el usuario no existe
        HTTPException 500: Si ocurre un error interno del servidor
    
    Example:
        PATCH /auth/users/5/role
        Body: {"new_role": "admin"}
    """
    try:
        # Validar ID de usuario
        if user_id <= 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="ID de usuario inválido"
            )
        
        # Validar que el rol sea válido
        if new_role not in ["admin", "user"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Rol inválido. Debe ser 'admin' o 'user'"
            )
        
        # Actualizar rol del usuario
        updated_user = AuthService.update_user_role(admin_user, user_id, new_role, db)
        
        # Retornar respuesta de confirmación
        return UserManagementResponse(
            message=f"Rol del usuario {updated_user.email} actualizado exitosamente a {new_role}",
            user_id=updated_user.id,
            user_email=updated_user.email,
            new_role=new_role,
            updated_by=admin_user.email,
            updated_at=datetime.utcnow()
        )
        
    except PermissionDeniedError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e)
        )


@router.patch("/users/{user_id}/deactivate", response_model=UserManagementResponse)
def deactivate_user(
    user_id: int,
    admin_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """
    Desactivar cuenta de usuario.
    
    Este endpoint permite a los administradores desactivar cuentas de usuario.
    Los usuarios desactivados no podrán iniciar sesión hasta ser reactivados.
    
    Restricciones:
        - Los administradores no pueden desactivarse a sí mismos
        - Los usuarios desactivados no pueden iniciar sesión
        - La desactivación no elimina los datos del usuario
    
    Args:
        user_id (int): ID del usuario a desactivar. Debe ser > 0.
        admin_user (User): Usuario administrador autenticado (inyectado automáticamente).
        db (Session): Sesión de base de datos (inyectada automáticamente).
    
    Returns:
        UserManagementResponse: Información de la operación incluyendo:
            - Mensaje de confirmación
            - ID del usuario desactivado
            - Email del usuario desactivado
            - Rol del usuario
            - Email del administrador que realizó la operación
            - Timestamp de la desactivación
    
    Raises:
        HTTPException 400: Si el user_id es inválido
        HTTPException 403: Si el usuario no tiene permisos o intenta desactivarse a sí mismo
        HTTPException 404: Si el usuario no existe
        HTTPException 500: Si ocurre un error interno del servidor
    
    Example:
        PATCH /auth/users/5/deactivate
    """
    try:
        # Validar ID de usuario
        if user_id <= 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="ID de usuario inválido"
            )
        
        # Desactivar usuario
        deactivated_user = AuthService.deactivate_user(admin_user, user_id, db)
        
        # Retornar respuesta de confirmación
        return UserManagementResponse(
            message=f"Usuario {deactivated_user.email} desactivado exitosamente",
            user_id=deactivated_user.id,
            user_email=deactivated_user.email,
            new_role=deactivated_user.role.name,
            updated_by=admin_user.email,
            updated_at=datetime.utcnow()
        )
        
    except PermissionDeniedError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e)
        )
    except UserNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unexpected error deactivating user: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error interno del servidor"
        )


@router.patch("/users/{user_id}/activate", response_model=UserManagementResponse)
def activate_user(
    user_id: int,
    admin_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """
    Activar cuenta de usuario.
    
    Este endpoint permite a los administradores reactivar cuentas de usuario
    previamente desactivadas. Los usuarios activados podrán iniciar sesión normalmente.
    
    Args:
        user_id (int): ID del usuario a activar. Debe ser > 0.
        admin_user (User): Usuario administrador autenticado (inyectado automáticamente).
        db (Session): Sesión de base de datos (inyectada automáticamente).
    
    Returns:
        UserManagementResponse: Información de la operación incluyendo:
            - Mensaje de confirmación
            - ID del usuario activado
            - Email del usuario activado
            - Rol del usuario
            - Email del administrador que realizó la operación
            - Timestamp de la activación
    
    Raises:
        HTTPException 400: Si el user_id es inválido
        HTTPException 403: Si el usuario no tiene privilegios de administrador
        HTTPException 404: Si el usuario no existe
        HTTPException 500: Si ocurre un error interno del servidor
    
    Example:
        PATCH /auth/users/5/activate
    """
    try:
        # Validar ID de usuario
        if user_id <= 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="ID de usuario inválido"
            )
        
        # Buscar usuario en base de datos
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Usuario no encontrado"
            )
        
        # Activar usuario
        user.is_active = True
        user.activated_at = datetime.utcnow()
        user.activated_by = admin_user.id
        db.commit()
        
        # Registrar operación en logs
        logger.info(f"User activated: {user.email} by {admin_user.email}")
        
        # Retornar respuesta de confirmación
        return UserManagementResponse(
            message=f"Usuario {user.email} activado exitosamente",
            user_id=user.id,
            user_email=user.email,
            new_role=user.role.name,
            updated_by=admin_user.email,
            updated_at=datetime.utcnow()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unexpected error activating user: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error interno del servidor"
        )


@router.get("/users/{user_id}/login-stats", response_model=LoginStatsResponse)
def get_user_login_stats(
    user_id: int,
    hours: int = 24,
    admin_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """
    Obtener estadísticas de intentos de inicio de sesión de un usuario.
    
    Este endpoint permite a los administradores consultar estadísticas detalladas
    sobre los intentos de inicio de sesión de un usuario específico en un período
    de tiempo determinado. Útil para análisis de seguridad y auditoría.
    
    Args:
        user_id (int): ID del usuario del cual obtener estadísticas. Debe ser > 0.
        hours (int): Número de horas hacia atrás para consultar. Por defecto 24 horas.
        admin_user (User): Usuario administrador autenticado (inyectado automáticamente).
        db (Session): Sesión de base de datos (inyectada automáticamente).
    
    Returns:
        LoginStatsResponse: Estadísticas de inicio de sesión incluyendo:
            - Email del usuario
            - Período de tiempo analizado (en horas)
            - Total de intentos de inicio de sesión
            - Intentos exitosos
            - Intentos fallidos
            - Tasa de éxito (porcentaje)
            - Fecha y hora del último intento
    
    Raises:
        HTTPException 400: Si el user_id es inválido
        HTTPException 403: Si el usuario no tiene privilegios de administrador
        HTTPException 404: Si el usuario no existe
        HTTPException 500: Si ocurre un error interno del servidor
    
    Example:
        GET /auth/users/5/login-stats?hours=48
    """
    try:
        # Validar ID de usuario
        if user_id <= 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="ID de usuario inválido"
            )
        
        # Verificar que el usuario existe
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Usuario no encontrado"
            )
        
        # Obtener estadísticas de inicio de sesión
        stats = AuthService.get_login_attempts_stats(user.email, db, hours)
        
        # Retornar estadísticas
        return LoginStatsResponse(
            email=stats["email"],
            period_hours=stats["period_hours"],
            total_attempts=stats["total_attempts"],
            successful_attempts=stats["successful_attempts"],
            failed_attempts=stats["failed_attempts"],
            success_rate=stats["success_rate"],
            last_attempt=stats["last_attempt"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error getting login stats for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error interno del servidor"
        )
