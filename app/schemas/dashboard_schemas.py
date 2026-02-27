
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

# =========================================================
# Type Breakdown
# =========================================================
class TypeBreakdown(BaseModel):
    """Desglose por tipo de archivo"""
    file_type: str = Field(..., description="Tipo de archivo (pdf, txt, docx)")
    count: int = Field(..., description="Número de documentos de este tipo")
    size: int = Field(..., description="Tamaño total en bytes")
    
    class Config:
        from_attributes = True

# =========================================================
# Esquemas para Funcionalidades de Dashboard
# =========================================================
class DashboardStats(BaseModel):
    """Estadísticas completas del dashboard"""
    totalDocuments: int = Field(..., description="Total de documentos")
    totalSize: int = Field(..., description="Tamaño total en bytes")
    documentsToday: int = Field(..., description="Documentos subidos hoy")
    documentsThisWeek: int = Field(..., description="Documentos subidos esta semana")
    documentsThisMonth: int = Field(..., description="Documentos subidos este mes")
    averagePerDay: float = Field(..., description="Promedio de documentos por día")
    mostActiveUser: Optional[str] = Field(None, description="Usuario más activo")
    peakUploadTime: Optional[str] = Field(None, description="Período con más subidas")
    typeBreakdown: List[TypeBreakdown] = Field(
        default_factory=list,
        description="Desglose de documentos por tipo"
    )  
    
    class Config:
        from_attributes = True

class ChartDataPoint(BaseModel):
    """Punto de datos para gráficos"""
    period: str = Field(..., alias="label", description="Período (fecha o etiqueta)")
    pdf: int = Field(0, description="Cantidad de PDFs")
    docx: int = Field(0, description="Cantidad de DOCX")
    txt: int = Field(0, description="Cantidad de TXT")
    total: int = Field(..., alias="value", description="Total de documentos")
    
    class Config:
        from_attributes = True
        populate_by_name = True
        
    @property
    def label(self) -> str:
        """Alias para compatibilidad con frontend"""
        return self.period
    
    @property
    def value(self) -> int:
        """Alias para compatibilidad con frontend"""
        return self.total


class ReporteItem(BaseModel):
    date: str
    pdf: int
    docx: int
    txt: int
    total: int

class UserStorageStats(BaseModel):
    """Estadísticas de almacenamiento por usuario"""
    user_id: int
    total_documents: int = Field(..., description="Total de documentos del usuario")
    total_size: int = Field(..., description="Tamaño total en bytes")
    type_breakdown: List[Dict[str, Any]] = Field(..., description="Desglose por tipo de archivo")
    
    class Config:
        from_attributes = True
        
class UserSummaryResponse(BaseModel):
    """Resumen completo del usuario"""
    user_id: int
    user_name: str
    user_email: str
    storage: UserStorageStats
    dashboard: DashboardStats
    recent_activities_count: int
    last_activity: Optional[datetime] = None
    generated_at: str
    
    class Config:
        from_attributes = True

# Esquema para métricas de rendimiento
class PerformanceMetrics(BaseModel):
    """Métricas de rendimiento del servicio"""
    avg_upload_time: Optional[float] = Field(None, description="Tiempo promedio de subida en segundos")
    avg_download_time: Optional[float] = Field(None, description="Tiempo promedio de descarga en segundos")
    success_rate: float = Field(..., description="Tasa de éxito de operaciones (0-100)")
    total_operations: int = Field(..., description="Total de operaciones realizadas")
    period_start: datetime = Field(..., description="Inicio del período de métricas")
    period_end: datetime = Field(..., description="Fin del período de métricas")
    
    class Config:
        from_attributes = True

# ========================================
#  ESQUEMAS PARA MÉTRICAS Y ANALYTICS
# ========================================

class UserActivityMetrics(BaseModel):
    """Métricas de actividad de usuario"""
    user_id: int
    login_frequency: dict  
    session_duration_avg: int  
    failed_login_attempts: int
    last_password_change: Optional[datetime]
    security_score: float  
    risk_level: str = Field(..., pattern="^(low|medium|high)$")

class SystemMetrics(BaseModel):
    """Métricas del sistema"""
    total_requests_24h: int
    successful_logins_24h: int
    failed_logins_24h: int
    new_registrations_24h: int
    active_sessions_current: int
    average_response_time_ms: float
    error_rate_percentage: float
    top_user_agents: List[dict]
    geographic_distribution: dict

class SecurityMetrics(BaseModel):
    """Métricas de seguridad"""
    suspicious_activities_24h: int
    blocked_ips_current: int
    failed_login_attempts_24h: int
    password_reset_requests_24h: int
    account_lockouts_24h: int
    security_incidents_open: int
    vulnerability_scan_last: Optional[datetime]
    compliance_score: float
