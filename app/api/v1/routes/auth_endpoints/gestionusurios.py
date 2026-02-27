"""
Módulo de gestión de perfil de usuario.

Este módulo contiene los endpoints relacionados con la gestión del perfil personal
del usuario autenticado, incluyendo consulta de información, cambio de contraseña
y estadísticas de inicio de sesión. Estos endpoints operan sobre el usuario actual
sin requerir privilegios de administrador.
"""

from app.services.security_service import verify_password
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from .....schemas.auth_schemas import (
    ActiveSessionsResponse, BackupCodesResponse, RefreshTokenRequest, 
    ResetPasswordRequest, Token, TwoFactorConfirmRequest, TwoFactorDisableRequest, 
    TwoFactorSetupResponse, TwoFactorVerifyRequest
)

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status, Body
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.models.models import User
from app.enums.enums import UserRole

from app.schemas.common_schemas import LoginStatsResponse
from app.schemas.user_schemas import UserCreate, UserInfoResponse, UserManagementResponse, UserUpdate
from app.services.auth_service import (
    AccountLockedError, AuthService, InvalidCredentialsError, PermissionDeniedError, 
    TokenBlacklistedError, TokenExpiredError, TwoFactorAuthService, UserAlreadyExistsError, 
    UserNotFoundError, WeakPasswordError, get_client_info, get_current_user, require_admin
)


# ========================================
# 🔧 CONFIGURACIÓN
# ========================================

router = APIRouter(prefix="/auth", tags=["authentication"])
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")
logger = logging.getLogger(__name__)


# ========================================
# 👤 ENDPOINTS DE GESTIÓN DE USUARIO
# ========================================

@router.get("/me", response_model=UserInfoResponse)
def get_current_user_info(current_user: User = Depends(get_current_user)):
    """
    Obtener información del perfil del usuario autenticado.
    
    Este endpoint retorna toda la información del perfil del usuario actualmente
    autenticado. Es el punto de entrada principal para que las aplicaciones cliente
    obtengan datos del usuario después del login.
    
    Información retornada:
        - Datos personales (ID, email, nombre)
        - Rol y permisos
        - Fechas relevantes (registro, último login)
        - Estado de la cuenta
        - Estado de autenticación de dos factores
    
    Args:
        current_user (User): Usuario autenticado actual (inyectado automáticamente)
    
    Returns:
        UserInfoResponse: Información completa del perfil incluyendo:
            - id: ID único del usuario
            - email: Dirección de correo electrónico
            - name: Nombre completo
            - role: Rol del usuario (admin, user)
            - created_at: Fecha de creación de la cuenta
            - last_login: Fecha y hora del último inicio de sesión
            - is_active: Si la cuenta está activa
            - two_factor_enabled: Si tiene 2FA habilitado
    
    Raises:
        HTTPException 401: Usuario no autenticado
        HTTPException 500: Error interno al procesar la solicitud
    
    Example:
        GET /auth/me
        Headers: Authorization: Bearer <access_token>
        
        Response:
        {
            "id": 1,
            "email": "usuario@example.com",
            "name": "Juan Pérez",
            "role": "user",
            "created_at": "2025-01-15T10:30:00Z",
            "last_login": "2025-11-02T20:00:00Z",
            "is_active": true,
            "two_factor_enabled": true
        }
    
    Notes:
        - Este endpoint no requiere permisos especiales más allá de la autenticación
        - La información retornada siempre corresponde al usuario del token
        - Los datos sensibles como contraseñas nunca se incluyen en la respuesta
    """
    try:
        # Construir respuesta con información del usuario
        return UserInfoResponse(
            id=current_user.id,
            email=current_user.email,
            name=current_user.name,
            role=current_user.role.name,
            created_at=current_user.created_at,
            last_login=current_user.last_login,
            is_active=current_user.is_active,
            two_factor_enabled=current_user.two_factor_enabled
        )
    except Exception as e:
        logger.exception(f"Error getting user info for user {current_user.id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error interno del servidor"
        )


@router.patch("/change-password", status_code=status.HTTP_200_OK, summary="Cambiar contraseña", description="Permite al usuario cambiar su contraseña actual")
def change_password(
    old_password: str = Body(..., embed=True, description="Current password"),
    new_password: str = Body(..., embed=True, description="New password"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Cambiar la contraseña del usuario autenticado.
    
    Este endpoint permite a los usuarios cambiar su propia contraseña. Requiere
    proporcionar la contraseña actual para verificación de identidad. La nueva
    contraseña debe cumplir con todos los requisitos de seguridad del sistema.
    
    Flujo de seguridad:
        1. Verificar que la contraseña actual es correcta
        2. Validar que la nueva contraseña cumple requisitos de seguridad
        3. Verificar que la nueva contraseña es diferente a la actual
        4. Actualizar contraseña en la base de datos
        5. Registrar el cambio en logs de auditoría
    
    Requisitos de seguridad para la nueva contraseña:
        - Mínimo 8 caracteres de longitud
        - Al menos una letra mayúscula (A-Z)
        - Al menos una letra minúscula (a-z)
        - Al menos un dígito numérico (0-9)
        - Al menos un carácter especial (!@#$%^&*()_+-=[]{}|;:,.<>?)
        - Debe ser diferente de la contraseña actual
    
    Args:
        old_password (str): Contraseña actual del usuario para verificación
        new_password (str): Nueva contraseña que cumple con requisitos de seguridad
        current_user (User): Usuario autenticado actual (inyectado automáticamente)
        db (Session): Sesión de base de datos (inyectada automáticamente)
    
    Returns:
        dict: Mensaje de confirmación:
            - message: "Contraseña cambiada exitosamente"
    
    Raises:
        HTTPException 400: Contraseña actual incorrecta o nueva contraseña no cumple requisitos
        HTTPException 401: Usuario no autenticado
        HTTPException 500: Error interno del servidor
    
    Example:
        PATCH /auth/change-password
        Headers: Authorization: Bearer <access_token>
        Body: {
            "old_password": "ContraseñaActual123!",
            "new_password": "NuevaContraseña456@"
        }
        
        Response:
        {
            "message": "Contraseña cambiada exitosamente"
        }
    
    Security Notes:
        - La contraseña actual debe coincidir exactamente
        - Los intentos fallidos se registran para detección de ataques
        - Las contraseñas se almacenan usando hash bcrypt con salt
        - No se permiten contraseñas comunes o débiles
        - El cambio de contraseña NO invalida tokens existentes automáticamente
    """
    try:
        # Cambiar contraseña usando el servicio de autenticación
        AuthService.change_password(current_user, old_password, new_password, db)
        
        # Registrar cambio exitoso en logs
        logger.info(f"Password changed successfully for user: {current_user.email}")
        return {"message": "Contraseña cambiada exitosamente"}
        
    except InvalidCredentialsError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except WeakPasswordError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unexpected error during password change for user {current_user.id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error interno del servidor"
        )


@router.get("/login-stats", response_model=LoginStatsResponse)
def get_login_stats(
    hours: int = 24,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Obtener estadísticas de intentos de inicio de sesión del usuario actual.
    
    Este endpoint permite a los usuarios consultar sus propias estadísticas de
    inicio de sesión para un período de tiempo específico. Útil para que los
    usuarios monitoreen la actividad de su cuenta y detecten accesos no autorizados.
    
    Métricas incluidas:
        - Total de intentos de inicio de sesión
        - Intentos exitosos vs. fallidos
        - Tasa de éxito (porcentaje)
        - Fecha y hora del último intento
    
    Casos de uso:
        - Revisar actividad reciente de la cuenta
        - Detectar intentos de acceso no autorizados
        - Verificar patrones de uso propios
        - Auditoría personal de seguridad
    
    Args:
        hours (int): Número de horas hacia atrás para analizar. Por defecto 24 horas.
            Valores comunes: 1, 6, 12, 24, 48, 168 (semana)
        current_user (User): Usuario autenticado actual (inyectado automáticamente)
        db (Session): Sesión de base de datos (inyectada automáticamente)
    
    Returns:
        LoginStatsResponse: Estadísticas de inicio de sesión incluyendo:
            - email: Email del usuario
            - period_hours: Período analizado en horas
            - total_attempts: Total de intentos de login
            - successful_attempts: Intentos exitosos
            - failed_attempts: Intentos fallidos
            - success_rate: Tasa de éxito en porcentaje (0-100)
            - last_attempt: Fecha/hora del último intento (puede ser None)
    
    Raises:
        HTTPException 401: Usuario no autenticado
        HTTPException 500: Error al procesar estadísticas
    
    Example:
        GET /auth/login-stats?hours=48
        Headers: Authorization: Bearer <access_token>
        
        Response:
        {
            "email": "usuario@example.com",
            "period_hours": 48,
            "total_attempts": 15,
            "successful_attempts": 14,
            "failed_attempts": 1,
            "success_rate": 93.33,
            "last_attempt": "2025-11-02T19:45:00Z"
        }
    
    Notes:
        - Solo muestra estadísticas del usuario autenticado
        - Los datos se calculan en tiempo real desde la base de datos
        - Incluye tanto inicios de sesión exitosos como fallidos
        - Los intentos bloqueados por 2FA también se cuentan
    """
    try:
        # Obtener estadísticas del servicio de autenticación
        stats = AuthService.get_login_attempts_stats(current_user.email, db, hours)
        
        # Construir respuesta con las estadísticas
        return LoginStatsResponse(
            email=stats["email"],
            period_hours=stats["period_hours"],
            total_attempts=stats["total_attempts"],
            successful_attempts=stats["successful_attempts"],
            failed_attempts=stats["failed_attempts"],
            success_rate=stats["success_rate"],
            last_attempt=stats["last_attempt"]
        )
        
    except Exception as e:
        logger.exception(f"Error getting login stats for user {current_user.id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error interno del servidor"
        )
