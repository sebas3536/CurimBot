"""
Router para autenticación de dos factores (2FA).

Este módulo implementa un sistema completo de autenticación de dos factores basado en TOTP
(Time-based One-Time Password) compatible con Google Authenticator, Microsoft Authenticator
y otras aplicaciones de autenticación estándar. Incluye generación de códigos de respaldo
para recuperación de cuenta en caso de pérdida del dispositivo.

Flujo de configuración 2FA:
    1. Usuario inicia setup en /2fa/setup
    2. Sistema genera secreto y código QR
    3. Usuario escanea QR con app de autenticación
    4. Usuario verifica código en /2fa/confirm
    5. Se generan y guardan códigos de respaldo
    6. 2FA queda habilitado

Security Features:
    - Códigos TOTP de 6 dígitos con expiración de 30 segundos
    - Códigos de respaldo de un solo uso para recuperación
    - Deshabilitación requiere código válido
    - Regeneración de códigos de respaldo con autorización
    - Tracking de cuándo fue habilitado/deshabilitado
    - Protección contra ataques de fuerza bruta
"""

import logging
from datetime import datetime
from .....schemas.auth_schemas import (
    BackupCodesResponse, TwoFactorConfirmRequest, TwoFactorDisableRequest, 
    TwoFactorSetupResponse, TwoFactorVerifyRequest
)
from fastapi import APIRouter, Depends, HTTPException, status, Body
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.models.models import User
from app.services.auth_service import TwoFactorAuthService, get_current_user


# ========================================
# 🔧 CONFIGURACIÓN
# ========================================

router = APIRouter(prefix="/auth", tags=["authentication"])
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")
logger = logging.getLogger(__name__)


# ========================================
# 🔐 ENDPOINTS DE AUTENTICACIÓN 2FA
# ========================================

@router.post("/2fa/setup", response_model=TwoFactorSetupResponse)
def setup_2fa(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Iniciar configuración de autenticación de dos factores.
    
    Este endpoint inicia el proceso de configuración de 2FA para el usuario.
    Genera un código QR que puede escanearse con Google Authenticator u otra
    aplicación TOTP compatible. También genera códigos de respaldo para
    recuperación de cuenta en emergencias.
    
    Flujo de operación:
        1. Verificar que 2FA no está ya habilitado
        2. Generar secreto TOTP (32 caracteres base32)
        3. Generar código QR con formato otpauth://
        4. Generar 10 códigos de respaldo de 8 dígitos
        5. Guardar secreto y códigos de forma temporal
        6. Retornar datos para que usuario verifique
    
    Requisitos:
        - Usuario autenticado
        - 2FA no debe estar ya habilitado
        - Primera vez completando este flujo (no tiene secreto pendiente)
    
    Información retornada:
        - **secret**: Clave secreta en Base32 (para entrada manual si QR no funciona)
        - **qr_code**: Código QR en formato Data URL (PNG base64)
        - **backup_codes**: Lista de 10 códigos de respaldo
        - **message**: Instrucciones para el usuario
    
    Args:
        current_user (User): Usuario autenticado actual (inyectado automáticamente)
        db (Session): Sesión de base de datos (inyectada automáticamente)
    
    Returns:
        TwoFactorSetupResponse: Información necesaria para configurar 2FA:
            - secret: Cadena Base32 para entrada manual (ej: "JBSWY3DPEBLW64TMMQ6AU...")
            - qr_code: Data URL de imagen PNG con código QR
            - backup_codes: Array de 10 códigos de 8 dígitos
            - message: Instrucción al usuario
    
    Raises:
        HTTPException 400: 2FA ya está habilitado
        HTTPException 401: Usuario no autenticado
        HTTPException 500: Error al generar secreto o QR
    
    Example:
        POST /auth/2fa/setup
        Headers: Authorization: Bearer <access_token>
        
        Response:
        {
            "secret": "JBSWY3DPEBLW64TMMQ6AU33SNKBXEJQ",
            "qr_code": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMIAAADC...",
            "backup_codes": [
                "12345678",
                "23456789",
                "34567890",
                ...
            ],
            "message": "Escanea el código QR con Google Authenticator y verifica con un código"
        }
    
    Security Notes:
        - El secreto debe guardarse SOLO en el dispositivo del usuario
        - Los códigos de respaldo DEBEN guardarse en lugar seguro offline
        - El QR contiene el secreto, se debe mostrar SOLO al usuario
        - La operación debe completarse dentro de 15 minutos
        - Si falla la confirmación, deben comenzar de nuevo
    
    Mobile Apps Soportadas:
        - Google Authenticator (iOS, Android)
        - Microsoft Authenticator (iOS, Android)
        - Authy (iOS, Android)
        - 1Password
        - Bitwarden
        - FreeOTP+
    """
    try:
        # Verificar si ya tiene 2FA habilitado
        if current_user.two_factor_enabled:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="2FA ya está habilitado. Desactívalo primero si quieres reconfigurarlo."
            )
        
        # Generar secreto y código QR
        secret, qr_code = TwoFactorAuthService.enable_2fa_for_user(current_user, db)
        
        # Generar códigos de respaldo
        backup_codes = TwoFactorAuthService.generate_backup_codes()
        TwoFactorAuthService.save_backup_codes(current_user, backup_codes, db)
        
        # Registrar intento de setup en logs
        logger.info(f"2FA setup initiated for user: {current_user.email}")
        
        return TwoFactorSetupResponse(
            secret=secret,
            qr_code=f"data:image/png;base64,{qr_code}",
            backup_codes=backup_codes,
            message="Escanea el código QR con Google Authenticator y verifica con un código"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error setting up 2FA for user {current_user.id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error al configurar 2FA"
        )


@router.post("/2fa/confirm", status_code=status.HTTP_200_OK)
def confirm_2fa_setup(
    data: TwoFactorConfirmRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Confirmar y completar configuración de autenticación de dos factores.
    
    Este endpoint verifica que la aplicación de autenticación está correctamente
    configurada validando un código TOTP. Si la verificación es exitosa, 2FA
    queda habilitado y el usuario deberá usar códigos cada vez que inicie sesión.
    
    Flujo de operación:
        1. Recibir código TOTP de 6 dígitos del usuario
        2. Validar que el código es correcto para el secreto generado
        3. Marcar 2FA como habilitado en base de datos
        4. Registrar timestamp de cuándo fue habilitado
        5. Guardar secreto de forma permanente
        6. Retornar confirmación al usuario
    
    Validaciones:
        - Código debe ser exactamente 6 dígitos
        - Código debe coincidir con el secreto guardado
        - Código no debe haber expirado (ventana de 30 segundos)
        - Debe estar en el flujo de setup (no debe tener 2FA ya habilitado)
    
    Args:
        data (TwoFactorConfirmRequest): Datos de confirmación:
            - code: Código TOTP de 6 dígitos
        current_user (User): Usuario autenticado actual (inyectado automáticamente)
        db (Session): Sesión de base de datos (inyectada automáticamente)
    
    Returns:
        dict: Confirmación de habilitación:
            - message: "2FA habilitado exitosamente"
            - enabled: True
            - enabled_at: Timestamp de cuándo se habilitó (ISO 8601)
    
    Raises:
        HTTPException 400: Código inválido o expirado
        HTTPException 401: Usuario no autenticado
        HTTPException 500: Error al confirmar 2FA
    
    Example:
        POST /auth/2fa/confirm
        Headers: Authorization: Bearer <access_token>
        Body: {
            "code": "123456"
        }
        
        Response:
        {
            "message": "2FA habilitado exitosamente",
            "enabled": true,
            "enabled_at": "2025-11-02T20:36:00.000Z"
        }
    
    Security Notes:
        - Después de este endpoint, todos los logins requieren 2FA
        - Los códigos son válidos por 30 segundos (ventana estándar TOTP)
        - Se permite margen de ±1 ventana de tiempo para reloj del servidor
        - Si falla, el usuario debe reintentar con otro código
        - Máximo 3 intentos fallidos antes de requerer nuevo setup
    
    Important:
        - El usuario debe guardar los códigos de respaldo ANTES de confirmar
        - Una vez confirmado, no se pueden recuperar los códigos originales
        - Los códigos de respaldo son CRÍTICOS para recuperación
    """
    try:
        # Verificar código y activar 2FA
        success = TwoFactorAuthService.confirm_2fa_setup(current_user, data.code, db)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Código inválido o expirado"
            )
        
        # Registrar confirmación en logs
        logger.info(f"2FA confirmed and enabled for user: {current_user.email}")
        
        return {
            "message": "2FA habilitado exitosamente",
            "enabled": True,
            "enabled_at": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.exception(f"Error confirming 2FA for user {current_user.id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error al confirmar 2FA"
        )


@router.post("/2fa/disable", status_code=status.HTTP_200_OK)
def disable_2fa(
    data: TwoFactorDisableRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Deshabilitar autenticación de dos factores.
    
    Este endpoint desactiva 2FA para el usuario. Requiere un código válido para
    autorizar la operación como medida de seguridad. El usuario puede usar un
    código TOTP actual o un código de respaldo.
    
    Casos de uso:
        - Usuario cambió a nuevo dispositivo de autenticación
        - Usuario perdió acceso a la app de autenticación
        - Usuario decidió no usar 2FA
        - Administrador deshabilitando 2FA por razones de seguridad
    
    Validaciones:
        - 2FA debe estar habilitado actualmente
        - Código debe ser válido (TOTP o backup)
        - Código TOTP: 6 dígitos válido en ventana actual
        - Código de respaldo: 8 dígitos no usado previamente
    
    Flujo de operación:
        1. Validar que 2FA está habilitado
        2. Intentar validar como código TOTP
        3. Si falla, intentar validar como código de respaldo
        4. Si ambos fallan, rechazar solicitud
        5. Si es válido, desabilitar 2FA:
           - Marcar two_factor_enabled como False
           - Limpiar secreto TOTP
           - Registrar timestamp de deshabilitación
        6. Invalidar todos los códigos de respaldo existentes
        7. Registrar la operación en logs de auditoría
    
    Args:
        data (TwoFactorDisableRequest): Datos de deshabilitación:
            - code: Código TOTP de 6 dígitos O código de respaldo de 8 dígitos
        current_user (User): Usuario autenticado actual (inyectado automáticamente)
        db (Session): Sesión de base de datos (inyectada automáticamente)
    
    Returns:
        dict: Confirmación de deshabilitación:
            - message: "2FA deshabilitado exitosamente"
            - enabled: False
            - disabled_at: Timestamp de cuándo se deshabilitó (ISO 8601)
    
    Raises:
        HTTPException 400: 2FA no está habilitado o código inválido
        HTTPException 401: Usuario no autenticado
        HTTPException 500: Error al deshabilitar 2FA
    
    Example 1 (con código TOTP):
        POST /auth/2fa/disable
        Headers: Authorization: Bearer <access_token>
        Body: {
            "code": "123456"
        }
        
        Response:
        {
            "message": "2FA deshabilitado exitosamente",
            "enabled": false,
            "disabled_at": "2025-11-02T20:36:00.000Z"
        }
    
    Example 2 (con código de respaldo):
        POST /auth/2fa/disable
        Headers: Authorization: Bearer <access_token>
        Body: {
            "code": "87654321"
        }
        
        Response:
        {
            "message": "2FA deshabilitado exitosamente",
            "enabled": false,
            "disabled_at": "2025-11-02T20:36:00.000Z"
        }
    
    Security Notes:
        - Requiere código válido para prevenir deshabilitación no autorizada
        - Si se usa código de respaldo, ese código se invalida inmediatamente
        - Todos los códigos de respaldo se invalidan al deshabilitar 2FA
        - El usuario seguirá autenticado en sesión actual
        - Próximo login NO requerirá 2FA
        - La operación se registra en logs de auditoría
        - Considerar notificar por email al usuario
    
    Advertencia:
        - Desabilitar 2FA reduce la seguridad de la cuenta
        - El usuario debería reconfigurar 2FA cuando sea posible
        - Si la cuenta fue comprometida, no solo deshabilitar 2FA
    """
    try:
        # Intentar primero con código TOTP
        success = False
        try:
            success = TwoFactorAuthService.disable_2fa_for_user(current_user, data.code, db)
        except ValueError:
            # Si falla TOTP, intentar con código de respaldo
            success = TwoFactorAuthService.verify_backup_code(current_user, data.code, db)
            if success:
                # Desabilitar 2FA
                current_user.two_factor_enabled = False
                current_user.two_factor_secret = None
                current_user.two_factor_disabled_at = datetime.utcnow()
                db.commit()
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Código inválido"
            )
        
        # Registrar deshabilitación en logs
        logger.info(f"2FA disabled for user: {current_user.email}")
        
        return {
            "message": "2FA deshabilitado exitosamente",
            "enabled": False,
            "disabled_at": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error disabling 2FA for user {current_user.id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error al deshabilitar 2FA"
        )


@router.get("/2fa/status", response_model=dict)
def get_2fa_status(
    current_user: User = Depends(get_current_user)
):
    """
    Obtener estado actual de autenticación de dos factores del usuario.
    
    Este endpoint retorna información sobre si 2FA está habilitado en la cuenta,
    cuándo fue habilitado y si hay códigos de respaldo disponibles. Útil para
    dashboards de seguridad y verificación de configuración.
    
    Información retornada:
        - **enabled**: Booleano indicando si 2FA está activo
        - **enabled_at**: Timestamp de cuándo fue habilitado (null si no activo)
        - **has_backup_codes**: Si existen códigos de respaldo guardados
    
    Casos de uso:
        - Widget de seguridad en dashboard
        - Verificación de estado antes de operaciones sensibles
        - Auditoría de configuración de seguridad
        - Avisos al usuario sobre estado de 2FA
    
    Args:
        current_user (User): Usuario autenticado actual (inyectado automáticamente)
    
    Returns:
        dict: Estado de 2FA del usuario:
            - enabled: True si está habilitado, False si no
            - enabled_at: Timestamp en ISO 8601 (null si no habilitado)
            - has_backup_codes: True si existen códigos de respaldo
    
    Raises:
        HTTPException 401: Usuario no autenticado
    
    Example 1 (2FA habilitado):
        GET /auth/2fa/status
        Headers: Authorization: Bearer <access_token>
        
        Response:
        {
            "enabled": true,
            "enabled_at": "2025-10-15T14:30:00Z",
            "has_backup_codes": true
        }
    
    Example 2 (2FA no habilitado):
        GET /auth/2fa/status
        Headers: Authorization: Bearer <access_token>
        
        Response:
        {
            "enabled": false,
            "enabled_at": null,
            "has_backup_codes": false
        }
    
    Notes:
        - No requiere ningún código o parámetro adicional
        - La respuesta es rápida (solo lectura de la BD)
        - Si enabled_at es null, 2FA no está configurado
        - Si has_backup_codes es false, usuario debería regenerar códigos
    """
    return {
        "enabled": current_user.two_factor_enabled,
        "enabled_at": current_user.two_factor_enabled_at.isoformat() if current_user.two_factor_enabled_at else None,
        "has_backup_codes": bool(current_user.backup_codes)
    }


@router.post("/2fa/regenerate-backup-codes", response_model=BackupCodesResponse)
def regenerate_backup_codes(
    code: str = Body(..., embed=True, description="Código 2FA para autorizar"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Regenerar códigos de respaldo para recuperación de 2FA.
    
    Este endpoint genera un nuevo conjunto de 10 códigos de respaldo, reemplazando
    los anteriores. Los códigos anteriores se invalidan inmediatamente. Requiere
    un código 2FA válido para autorizar la operación como medida de seguridad.
    
    Casos de uso:
        - Usuario comprometió sus códigos de respaldo
        - Usuario perdió sus códigos de respaldo
        - Usuario quiere actualizar códigos por política de seguridad
        - Usuario accidentalmente usó todos los códigos
    
    Validaciones:
        - 2FA debe estar habilitado
        - Código debe ser válido (código TOTP actual)
        - Usuario debe estar autenticado
    
    Flujo de operación:
        1. Verificar que 2FA está habilitado
        2. Validar código TOTP proporcionado
        3. Si es inválido, rechazar solicitud
        4. Generar 10 nuevos códigos de respaldo
        5. Invalidar códigos anteriores
        6. Guardar nuevos códigos en BD
        7. Retornar códigos al usuario
        8. Registrar la operación en logs
    
    Args:
        code (str): Código TOTP de 6 dígitos para autorizar la operación
        current_user (User): Usuario autenticado actual (inyectado automáticamente)
        db (Session): Sesión de base de datos (inyectada automáticamente)
    
    Returns:
        BackupCodesResponse: Nuevos códigos de respaldo:
            - backup_codes: Array de 10 códigos de 8 dígitos
            - message: Instrucción de guardar en lugar seguro
    
    Raises:
        HTTPException 400: 2FA no habilitado o código inválido
        HTTPException 401: Usuario no autenticado
        HTTPException 500: Error al regenerar códigos
    
    Example:
        POST /auth/2fa/regenerate-backup-codes
        Headers: Authorization: Bearer <access_token>
        Body: {
            "code": "123456"
        }
        
        Response:
        {
            "backup_codes": [
                "12345678",
                "23456789",
                "34567890",
                "45678901",
                "56789012",
                "67890123",
                "78901234",
                "89012345",
                "90123456",
                "01234567"
            ],
            "message": "Códigos de respaldo regenerados. Guárdalos en un lugar seguro."
        }
    
    Security Notes:
        - Requiere código TOTP para autorizar (previene regeneración no autorizada)
        - Los códigos anteriores se invalidan INMEDIATAMENTE
        - Los nuevos códigos son de un solo uso cada uno
        - CRÍTICO: Usuario debe guardar los nuevos códigos offline
        - Si usuario pierde acceso a app 2FA, los códigos de respaldo son su ÚNICA opción
        - Máximo 10 códigos simultáneamente disponibles
        - Cada código solo puede usarse una vez
    
    Best Practices:
        - Guardar códigos en lugar seguro (password manager, papel, etc.)
        - Regenerar cuando se comprometan los códigos
        - Considerar regenerar cada 6 meses por política
        - No compartir códigos con nadie
        - Verificar que usuario tiene nuevos códigos antes de finalizar
    """
    try:
        # Verificar que 2FA está habilitado
        if not current_user.two_factor_enabled:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="2FA no está habilitado"
            )
        
        # Verificar código TOTP para autorizar
        if not TwoFactorAuthService.verify_totp_code(current_user.two_factor_secret, code):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Código inválido"
            )
        
        # Generar nuevos códigos de respaldo
        backup_codes = TwoFactorAuthService.generate_backup_codes()
        TwoFactorAuthService.save_backup_codes(current_user, backup_codes, db)
        
        # Registrar regeneración en logs
        logger.info(f"Backup codes regenerated for user: {current_user.email}")
        
        return BackupCodesResponse(
            backup_codes=backup_codes,
            message="Códigos de respaldo regenerados. Guárdalos en un lugar seguro."
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error regenerating backup codes for user {current_user.id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error al regenerar códigos de respaldo"
        )
