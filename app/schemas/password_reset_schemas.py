"""
Schemas para recuperación de contraseña
app/schemas/password_reset_schemas.py
"""
from pydantic import BaseModel, EmailStr, Field, field_validator
import re


class PasswordResetRequest(BaseModel):
    """Schema para solicitud de recuperación de contraseña"""
    email: EmailStr = Field(..., description="Email del usuario")


class PasswordResetVerify(BaseModel):
    """Schema para verificar token de recuperación"""
    token: str = Field(..., min_length=32, description="Token de recuperación")


class PasswordResetConfirm(BaseModel):
    """Schema para confirmar nueva contraseña"""
    token: str = Field(..., min_length=32, description="Token de recuperación")
    new_password: str = Field(
        ...,
        min_length=8,
        max_length=100,
        description="Nueva contraseña (mínimo 8 caracteres)"
    )
    
    @field_validator('new_password')
    @classmethod
    def validate_password_strength(cls, v: str) -> str:
        """Valida la fortaleza de la contraseña"""
        if len(v) < 8:
            raise ValueError('La contraseña debe tener al menos 8 caracteres')
        
        if not re.search(r'[A-Z]', v):
            raise ValueError('La contraseña debe contener al menos una mayúscula')
        
        if not re.search(r'[a-z]', v):
            raise ValueError('La contraseña debe contener al menos una minúscula')
        
        if not re.search(r'\d', v):
            raise ValueError('La contraseña debe contener al menos un número')
        
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', v):
            raise ValueError('La contraseña debe contener al menos un carácter especial')
        
        return v


class PasswordResetResponse(BaseModel):
    """Schema para respuesta de operaciones de recuperación"""
    message: str = Field(..., description="Mensaje de respuesta")


class TokenValidationResponse(BaseModel):
    """Schema para respuesta de validación de token"""
    valid: bool = Field(..., description="Si el token es válido")
    message: str = Field(..., description="Mensaje descriptivo")