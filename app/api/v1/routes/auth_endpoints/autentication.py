"""
Módulo de autenticación y seguridad de usuarios.

Este módulo contiene los endpoints relacionados con la autenticación de usuarios,
incluyendo login/logout, registro, renovación de tokens, autenticación de dos factores (2FA)
y gestión de alertas de inicio de sesión. Implementa OAuth2 y TOTP para máxima seguridad.
"""

from fastapi.responses import JSONResponse
from app.services import security_service
from app.services.security_service import verify_password
import logging
from datetime import datetime
from typing import List, Optional
from .....schemas.auth_schemas import (
    ActiveSessionsResponse, BackupCodesResponse, RefreshTokenRequest, 
    ResetPasswordRequest, Token, TwoFactorConfirmRequest, TwoFactorDisableRequest, 
    TwoFactorSetupResponse, TwoFactorVerifyRequest
)

from fastapi import APIRouter, Depends, Form, HTTPException, Request, Response, status, Body
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.models.models import User, LoginAlert
from app.enums.enums import UserRole

from app.schemas.common_schemas import LoginStatsResponse
from app.schemas.user_schemas import UserCreate, UserInfoResponse, UserManagementResponse
from app.services.auth_service import (
    AccountLockedError, AuthService, InvalidCredentialsError, PermissionDeniedError, 
    TokenBlacklistedError, TokenExpiredError, TwoFactorAuthService, TwoFactorRequiredError, 
    UserAlreadyExistsError, UserNotFoundError, WeakPasswordError, get_client_info, 
    get_current_user, require_admin
)

from app.services.login_alert_service import LoginAlertService
from app.services.email_service import EmailService

# Importar configuración
from app.core.config import RESEND_API_KEY, FROM_EMAIL


# ========================================
# 🔧 CONFIGURACIÓN
# ========================================

router = APIRouter(prefix="/auth", tags=["authentication"])
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")
logger = logging.getLogger(__name__)


# ========================================
# 🔐 ENDPOINTS DE AUTENTICACIÓN BÁSICA
# ========================================

@router.post("/login", response_model=Token, summary="Iniciar sesión", description="Autentica usuario y retorna tokens de acceso")
def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    request: Request = None,
    db: Session = Depends(get_db)
):
    """
    Autenticar usuario y obtener tokens de acceso.
    
    Este endpoint implementa el flujo de autenticación OAuth2 estándar. Verifica las
    credenciales del usuario y retorna tokens de acceso y renovación. Si el usuario
    tiene autenticación de dos factores habilitada, retorna una respuesta especial
    indicando que se requiere verificación 2FA.
    
    Características de seguridad:
        - Bloqueo de cuenta después de 5 intentos fallidos
        - Seguimiento de intentos de inicio de sesión
        - Registro de dirección IP y User-Agent
        - Soporte para autenticación de dos factores
    
    Args:
        form_data (OAuth2PasswordRequestForm): Credenciales del usuario incluyendo:
            - username: Dirección de email del usuario
            - password: Contraseña del usuario
        request (Request, opcional): Objeto Request de FastAPI para obtener información del cliente
        db (Session): Sesión de base de datos (inyectada automáticamente)
    
    Returns:
        Token: Tokens de autenticación incluyendo:
            - access_token: Token JWT para acceso a recursos protegidos
            - refresh_token: Token para renovar el access_token
            - token_type: Tipo de token (siempre "bearer")
        
        O JSONResponse si requiere 2FA:
            - requires_2fa: True indicando que se necesita código 2FA
            - message: Mensaje descriptivo
    
    Raises:
        HTTPException 401: Credenciales inválidas
        HTTPException 423: Cuenta bloqueada por múltiples intentos fallidos
        HTTPException 500: Error interno del servidor
    
    Example:
        POST /auth/login
        Form Data: username=user@example.com&password=SecurePass123!
    """
    try:
        # Obtener información del cliente (IP y User-Agent)
        ip_address, user_agent = get_client_info(request) if request else ("unknown", "unknown")
        
        # Intentar autenticar usuario
        result = AuthService.login_user(
            form_data.username,
            form_data.password,
            db,
            ip_address,
            user_agent
        )
        
        # Verificar si requiere autenticación de dos factores
        if isinstance(result, dict) and result.get("requires_2fa"):
            return JSONResponse(
                status_code=200,  
                content={
                    "requires_2fa": True,
                    "message": "Verificación de dos factores requerida"
                }
            )
        
        # Si no requiere 2FA, result es tupla (access_token, refresh_token)
        access_token, refresh_token = result
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer"
        }
        
    except InvalidCredentialsError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Credenciales inválidas"
        )
    
    except AccountLockedError:
        raise HTTPException(
            status_code=status.HTTP_423_LOCKED,
            detail="Cuenta bloqueada"
        )
    
    except HTTPException:
        raise
    
    except Exception as e:
        logger.exception(f"Unexpected error during login: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error interno del servidor durante el login"
        )


@router.post("/signup", response_model=Token, status_code=status.HTTP_201_CREATED, summary="Registrar nuevo usuario", description="Crea una nueva cuenta de usuario y retorna tokens de autenticación")
def signup(
    user_data: UserCreate,
    request: Request,
    db: Session = Depends(get_db)
):
    """
    Registrar una nueva cuenta de usuario en el sistema.
    
    Este endpoint permite crear nuevas cuentas de usuario. La contraseña debe cumplir
    con requisitos de seguridad estrictos. Después del registro exitoso, el usuario
    recibe tokens de acceso automáticamente para iniciar sesión inmediatamente.
    
    Requisitos de contraseña:
        - Mínimo 8 caracteres
        - Al menos una letra mayúscula
        - Al menos una letra minúscula
        - Al menos un dígito
        - Al menos un carácter especial
    
    Args:
        user_data (UserCreate): Datos del nuevo usuario incluyendo:
            - name: Nombre completo del usuario
            - email: Dirección de email (se usará para iniciar sesión)
            - password: Contraseña que cumpla con los requisitos de seguridad
            - password_confirm: Confirmación de contraseña (debe coincidir)
        request (Request): Objeto Request de FastAPI para obtener dirección IP
        db (Session): Sesión de base de datos (inyectada automáticamente)
    
    Returns:
        Token: Tokens de autenticación incluyendo:
            - access_token: Token JWT para acceso inmediato
            - refresh_token: Token para renovar el access_token
            - token_type: Tipo de token (siempre "bearer")
    
    Raises:
        HTTPException 400: Contraseña no cumple requisitos de seguridad
        HTTPException 409: Usuario ya existe con ese email
        HTTPException 500: Error interno del servidor
    
    Example:
        POST /auth/signup
        Body: {
            "name": "Juan Pérez",
            "email": "juan@example.com",
            "password": "SecurePass123!",
            "password_confirm": "SecurePass123!"
        }
    """
    try:
        # Obtener dirección IP del cliente
        ip_address, _ = get_client_info(request)
        
        # Crear nuevo usuario y obtener tokens
        access_token, refresh_token = AuthService.signup_user(user_data, db, ip_address)
        
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer"
        }
        
    except UserAlreadyExistsError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
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
        logger.exception(f"Unexpected error during signup: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error interno del servidor durante el registro"
        )


@router.post("/logout", status_code=status.HTTP_200_OK, summary="Cerrar sesión", description="Invalida el token actual del usuario")
def logout(
    response: Response,
    token: str = Depends(oauth2_scheme),
    data: RefreshTokenRequest = Body(...), 
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Cerrar sesión del usuario actual e invalidar tokens.
    
    Este endpoint invalida el refresh token del usuario, agregándolo a una lista negra
    para prevenir su reutilización. El token de acceso continuará siendo válido hasta
    su expiración natural, pero no se podrá usar el refresh token para obtener nuevos
    tokens de acceso.
    
    Args:
        response (Response): Objeto Response de FastAPI (inyectado automáticamente)
        token (str): Token de acceso actual (extraído del header Authorization)
        data (RefreshTokenRequest): Objeto conteniendo:
            - refresh_token: Token de renovación a invalidar
        current_user (User): Usuario autenticado actual (inyectado automáticamente)
        db (Session): Sesión de base de datos (inyectada automáticamente)
    
    Returns:
        dict: Mensaje de confirmación:
            - message: "Sesión cerrada correctamente"
    
    Raises:
        HTTPException 401: Token ya estaba invalidado previamente
        HTTPException 500: Error interno del servidor
    
    Example:
        POST /auth/logout
        Headers: Authorization: Bearer <access_token>
        Body: {"refresh_token": "<refresh_token>"}
    """
    try:
        # Invalidar refresh token agregándolo a lista negra
        was_blacklisted = AuthService.logout_user(data.refresh_token, db)
        message = "Sesión cerrada correctamente"
        
        if not was_blacklisted:
            raise HTTPException(status_code=401, detail="Token ya estaba invalidado")
        
        # Registrar cierre de sesión en logs
        logger.info(f"User logged out: {current_user.email}")
        return {"message": message}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unexpected error during logout: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error interno del servidor durante el logout"
        )


@router.post("/refresh", response_model=Token, summary="Renovar tokens", description="Renueva tokens de acceso usando un refresh token válido")
def refresh_token(
    data: RefreshTokenRequest,
    db: Session = Depends(get_db)
):
    """
    Renovar tokens de acceso usando un refresh token válido.
    
    Este endpoint permite obtener un nuevo par de tokens (access_token y refresh_token)
    usando un refresh token válido. El refresh token antiguo se invalida automáticamente
    para prevenir su reutilización (protección contra ataques de replay).
    
    Flujo de renovación:
        1. Validar que el refresh token no esté en lista negra
        2. Verificar que el refresh token no haya expirado
        3. Extraer información del usuario del token
        4. Generar nuevo par de tokens
        5. Invalidar el refresh token antiguo
    
    Args:
        data (RefreshTokenRequest): Objeto conteniendo:
            - refresh_token: Token de renovación válido
        db (Session): Sesión de base de datos (inyectada automáticamente)
    
    Returns:
        Token: Nuevos tokens de autenticación incluyendo:
            - access_token: Nuevo token JWT para acceso a recursos
            - refresh_token: Nuevo token para futuras renovaciones
            - token_type: Tipo de token (siempre "bearer")
    
    Raises:
        HTTPException 401: Token expirado o en lista negra
        HTTPException 404: Usuario no encontrado
        HTTPException 500: Error interno del servidor
    
    Example:
        POST /auth/refresh
        Body: {"refresh_token": "<valid_refresh_token>"}
    """
    try:
        # Renovar tokens y obtener nuevo par
        access_token, refresh_token = AuthService.refresh_tokens(data.refresh_token, db)
        
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer"
        }
        
    except (TokenExpiredError, TokenBlacklistedError) as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
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
        logger.exception(f"Unexpected error during token refresh: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error interno del servidor durante la renovación de tokens"
        )


# ========================================
# 🔐 ENDPOINTS DE AUTENTICACIÓN CON 2FA
# ========================================

@router.post("/login-with-2fa", response_model=Token)
def login_with_2fa(
    request: Request,
    form_data: OAuth2PasswordRequestForm = Depends(),
    totp_code: str = Form(...),
    db: Session = Depends(get_db)
):
    """
    Iniciar sesión con autenticación de dos factores (2FA).
    
    Este endpoint implementa el flujo completo de autenticación con verificación
    de dos factores utilizando códigos TOTP (Time-based One-Time Password).
    También soporta códigos de respaldo en caso de que el usuario no tenga
    acceso a su aplicación de autenticación.
    
    Flujo de autenticación:
        1. Verificar credenciales básicas (email y contraseña)
        2. Validar código TOTP de 6 dígitos o código de respaldo
        3. Generar tokens de acceso y renovación
        4. Actualizar timestamp de último inicio de sesión
        5. Registrar evento de login y verificar si es sospechoso
        6. Enviar alerta por email si el login es desde dispositivo/ubicación nueva
    
    Args:
        request (Request): Objeto Request de FastAPI para obtener IP y User-Agent
        form_data (OAuth2PasswordRequestForm): Credenciales del usuario:
            - username: Dirección de email del usuario
            - password: Contraseña del usuario
        totp_code (str): Código de 6 dígitos de la aplicación 2FA o código de respaldo
        db (Session): Sesión de base de datos (inyectada automáticamente)
    
    Returns:
        Token: Tokens de autenticación incluyendo:
            - access_token: Token JWT para acceso a recursos protegidos
            - refresh_token: Token para renovar el access_token
            - token_type: Tipo de token (siempre "bearer")
    
    Raises:
        HTTPException 401: Credenciales inválidas o código 2FA inválido/expirado
        HTTPException 500: Error interno del servidor
    
    Example:
        POST /auth/login-with-2fa
        Form Data: username=user@example.com&password=SecurePass123!&totp_code=123456
    
    Notes:
        - Los códigos TOTP expiran después de 30 segundos
        - Los códigos de respaldo solo se pueden usar una vez
        - Se registran todos los intentos de login para análisis de seguridad
    """
    try:
        # 1. Autenticar usuario con credenciales básicas
        user = db.query(User).filter(User.email == form_data.username).first()
        if not user or not verify_password(form_data.password, user.password_hash):
            raise HTTPException(status_code=401, detail="Credenciales inválidas")
        
        # 2. Verificar código 2FA (TOTP o backup code)
        if not TwoFactorAuthService.verify_totp_code(user.two_factor_secret, totp_code):
            # Intentar con código de respaldo si TOTP falla
            if not TwoFactorAuthService.verify_backup_code(user, totp_code, db):
                raise HTTPException(status_code=401, detail="Código inválido")
        
        # 3. Generar tokens de acceso
        token_data = {
            "sub": str(user.id),
            "role": user.role.name,
            "email": user.email
        }
        access_token = security_service.create_access_token(token_data)
        refresh_token = security_service.create_refresh_token(token_data)
        
        # 4. Actualizar timestamp de último login
        AuthService.update_last_login(user, db)
        
        # 5. Registrar login y verificar si se debe enviar alerta
        try:
            email_service = EmailService(api_key=RESEND_API_KEY, from_email=FROM_EMAIL)
            alert_service = LoginAlertService(email_service)
            alert_service.record_login_and_check(user, request, db)
        except Exception as alert_error:
            # Registrar error pero no fallar el login por esto
            logger.warning(f"Failed to record login alert: {alert_error}")
        
        # Registrar login exitoso en logs
        logger.info(f"Successful login for user: {user.email}")
        
        return Token(access_token=access_token, refresh_token=refresh_token)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error during login: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error al procesar inicio de sesión"
        )


# ========================================
# 🔔 ENDPOINTS DE ALERTAS DE LOGIN
# ========================================

@router.get("/login-alerts/recent")
def get_recent_login_alerts(
    days: int = 30,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Obtener alertas de inicio de sesión recientes del usuario autenticado.
    
    Este endpoint permite a los usuarios consultar sus alertas de inicio de sesión
    recientes para detectar actividad sospechosa o accesos no autorizados. Las
    alertas incluyen información sobre dispositivos nuevos, ubicaciones nuevas
    y eventos marcados como sospechosos.
    
    Tipos de alertas:
        - **Dispositivo nuevo**: Login desde un dispositivo no reconocido
        - **Ubicación nueva**: Login desde una ubicación geográfica diferente
        - **Actividad sospechosa**: Patrones de acceso inusuales
    
    Args:
        days (int): Número de días hacia atrás para consultar. Por defecto 30 días.
        current_user (User): Usuario autenticado actual (inyectado automáticamente)
        db (Session): Sesión de base de datos (inyectada automáticamente)
    
    Returns:
        dict: Resumen de alertas incluyendo:
            - total: Número total de alertas en el período
            - alerts: Lista de alertas, cada una con:
                - id: ID único de la alerta
                - device: Información del dispositivo (User-Agent)
                - location: Ubicación geográfica estimada
                - ip_address: Dirección IP del acceso
                - is_suspicious: Si fue marcado como sospechoso
                - is_new_device: Si es un dispositivo nuevo
                - is_new_location: Si es una ubicación nueva
                - created_at: Fecha y hora del evento (formato ISO 8601)
    
    Raises:
        HTTPException 401: Usuario no autenticado
        HTTPException 500: Error interno del servidor
    
    Example:
        GET /auth/login-alerts/recent?days=7
        Headers: Authorization: Bearer <access_token>
    """
    try:
        # Obtener alertas recientes del usuario
        alerts = LoginAlertService.get_recent_login_alerts(
            user=current_user,
            days=days,
            db=db
        )
        
        # Formatear respuesta con información de cada alerta
        return {
            "total": len(alerts),
            "alerts": [
                {
                    "id": alert.id,
                    "device": alert.device,
                    "location": alert.location,
                    "ip_address": alert.ip_address,
                    "is_suspicious": alert.is_suspicious,
                    "is_new_device": alert.is_new_device,
                    "is_new_location": alert.is_new_location,
                    "created_at": alert.created_at.isoformat()
                }
                for alert in alerts
            ]
        }
    except Exception as e:
        logger.exception(f"Error getting login alerts for user {current_user.id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error al obtener alertas de login"
        )


@router.delete("/login-alerts/{alert_id}")
def dismiss_login_alert(
    alert_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Descartar una alerta de inicio de sesión específica.
    
    Este endpoint permite a los usuarios eliminar alertas de inicio de sesión
    que ya han revisado o que reconocen como accesos legítimos propios.
    Solo se pueden eliminar alertas que pertenezcan al usuario autenticado.
    
    Args:
        alert_id (int): ID único de la alerta a descartar
        current_user (User): Usuario autenticado actual (inyectado automáticamente)
        db (Session): Sesión de base de datos (inyectada automáticamente)
    
    Returns:
        dict: Mensaje de confirmación:
            - message: "Alerta descartada exitosamente"
    
    Raises:
        HTTPException 401: Usuario no autenticado
        HTTPException 404: Alerta no encontrada o no pertenece al usuario
        HTTPException 500: Error interno del servidor
    
    Example:
        DELETE /auth/login-alerts/123
        Headers: Authorization: Bearer <access_token>
    
    Security:
        - Los usuarios solo pueden eliminar sus propias alertas
        - El ID de alerta debe existir y estar asociado al usuario autenticado
    """
    try:
        # Buscar alerta que pertenezca al usuario autenticado
        alert = db.query(LoginAlert).filter(
            LoginAlert.id == alert_id,
            LoginAlert.user_id == current_user.id
        ).first()
        
        # Verificar que la alerta existe y pertenece al usuario
        if not alert:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Alerta no encontrada"
            )
        
        # Eliminar alerta de la base de datos
        db.delete(alert)
        db.commit()
        
        return {"message": "Alerta descartada exitosamente"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error dismissing login alert {alert_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error al descartar alerta"
        )
