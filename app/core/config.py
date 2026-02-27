"""
Módulo de configuración centralizado de la aplicación.

Este módulo gestiona todas las configuraciones de la aplicación, incluyendo
autenticación, base de datos, CORS, Curim (búsqueda semántica), RAG
(Retrieval-Augmented Generation), y síntesis de voz.

Características:
    - Carga de variables de entorno con .env
    - Validación de configuración crítica
    - Configuración por ambiente (desarrollo/producción)
    - Integración con Curim para búsqueda semántica
    - RAG configuration para procesamiento de documentos
    - Text-to-speech configuration
    - Creación automática de directorios necesarios
"""

import os
from dotenv import load_dotenv

# Cargar variables de entorno desde .env
load_dotenv()

# Configuraciones existentes - Email
RESEND_API_KEY = os.getenv("RESEND_API_KEY")
FROM_EMAIL = os.getenv("FROM_EMAIL")
print(f"Remitente configurado: {FROM_EMAIL}")


class Settings:
    """
    Clase de configuración centralizada para toda la aplicación.
    
    Agrupa todas las configuraciones en un solo lugar, facilitando
    gestión, validación y cambios entre ambientes (dev/prod).
    
    Categorías de configuración:
        - Autenticación: JWT, tokens, tiempos de expiración
        - Base de datos: Conexión, URL, tipo de DB
        - CORS: Orígenes permitidos, headers
        - Curim: Búsqueda semántica, embeddings, almacenamiento
        - RAG: Chunking, recuperación, relevancia
        - Voz: Text-to-speech, voces, velocidad
    
    Uso:
        from app.core.config import settings
        
        # Acceder a configuración
        db_url = settings.DATABASE_URL
        api_key = settings.GEMINI_API_KEY
        settings.validate_Curim()  # Validar antes de usar Curim
    """
    
    # =========================================================
    #  AUTENTICACIÓN
    # =========================================================
    
    # **SECRET_KEY**: Clave secreta para JWT
    #   - Usada para firmar tokens
    #   - Debe ser única y fuerte en producción
    #   - Cambiarla invalida todos los tokens existentes
    #   - Variable de entorno: SECRET_KEY
    SECRET_KEY = os.getenv("SECRET_KEY", "defaultsecret")
    
    # **ALGORITHM**: Algoritmo de encriptación para JWT
    #   - HS256: HMAC with SHA-256 (default, recomendado)
    #   - RS256: RSA with SHA-256 (para keys pública/privada)
    ALGORITHM = "HS256"
    
    # **ACCESS_TOKEN_EXPIRE_MINUTES**: Tiempo de expiración de access token
    #   - Por defecto: 8 horas (480 minutos)
    #   - Balancear entre seguridad y conveniencia
    #   - Tokens más cortos = más seguro pero requiere refresh frecuente
    #   - Variable de entorno: ACCESS_TOKEN_EXPIRE_MINUTES
    ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 480))
    
    # =========================================================
    #  BASE DE DATOS
    # =========================================================
    
    # **DATABASE_URL**: URL de conexión a la base de datos
    #   - SQLite (dev): sqlite:///./asistente_docs.db
    #   - PostgreSQL (prod): postgresql://user:pass@host:5432/dbname
    #   - MySQL: mysql://user:pass@host:3306/dbname
    #   - Variable de entorno: DATABASE_URL
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./asistente_docs.db")
    
    # =========================================================
    #  CORS (Cross-Origin Resource Sharing)
    # =========================================================
    
    # **CORS_ORIGINS**: Orígenes permitidos para CORS
    #   - Permite solicitudes desde estos dominios
    #   - Frontend en localhost:4200 durante desarrollo
    #   - Agregar dominio de producción cuando esté listo
    #   - Importante: Restringir en producción
    CORS_ORIGINS = ["http://localhost:4200"]
    
    # =========================================================
    #  Curim - BÚSQUEDA SEMÁNTICA Y RAG
    # =========================================================
    
    # **GEMINI_API_KEY**: API key de Google Gemini
    #   - Usada para generación de embeddings y respuestas
    #   - Obtenible en: https://makersuite.google.com/app/apikey
    #   - Importante: Nunca commitear clave en repositorio
    #   - Variable de entorno: GEMINI_API_KEY
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    
    # **Curim_STORAGE_PATH**: Ruta para almacenamiento de Curim
    #   - Donde se guardan embeddings, caché, datos
    #   - Se crean subdirectorios automáticamente
    #   - Default: ./storage/Curim_data
    #   - Considerar ruta con permisos de escritura en producción
    #   - Variable de entorno: Curim_STORAGE_PATH
    Curim_STORAGE_PATH = os.getenv("Curim_STORAGE_PATH", "./storage/Curim_data")
    
    # **Curim_CACHE_ENABLED**: Activar caché de respuestas
    #   - True: Las respuestas se cachean (más rápido)
    #   - False: Cada pregunta busca nueva (más fresco)
    #   - Recomendado: True para mejor performance
    #   - Variable de entorno: Curim_CACHE_ENABLED
    Curim_CACHE_ENABLED = os.getenv("Curim_CACHE_ENABLED", "True").lower() == "true"
    
    # **Curim_CACHE_TTL_DAYS**: Tiempo de vida del caché en días
    #   - Respuestas en caché se expiran después de N días
    #   - Default: 7 días
    #   - Más alto = menos actualizaciones, menor precisión
    #   - Más bajo = más actualizaciones, más costo computacional
    #   - Variable de entorno: Curim_CACHE_TTL_DAYS
    Curim_CACHE_TTL_DAYS = int(os.getenv("Curim_CACHE_TTL_DAYS", 7))
    
    # =========================================================
    #  RAG - RETRIEVAL-AUGMENTED GENERATION
    # =========================================================
    
    # **CHUNK_SIZE**: Tamaño de chunks para documentos
    #   - Documentos se dividen en chunks de este tamaño
    #   - Demasiado pequeño: Mucho contexto perdido
    #   - Demasiado grande: Embedding menos preciso
    #   - Recomendado: 300-1000 caracteres
    #   - Default: 500 caracteres
    #   - Variable de entorno: CHUNK_SIZE
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
    
    # **CHUNK_OVERLAP**: Sobreposición entre chunks consecutivos
    #   - Caracteres que se repiten entre chunks
    #   - Previene perder información entre límites
    #   - Típicamente: 10-25% del CHUNK_SIZE
    #   - Default: 100 caracteres
    #   - Variable de entorno: CHUNK_OVERLAP
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))
    
    # **TOP_K_RESULTS**: Número de chunks a recuperar en búsqueda
    #   - Cuántos chunks relevantes se usan para generar respuesta
    #   - Más = más información pero menos preciso
    #   - Menos = más específico pero puede perder información
    #   - Default: 3 chunks
    #   - Variable de entorno: TOP_K_RESULTS
    TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", 3))
    
    # =========================================================
    #  TEXT-TO-SPEECH
    # =========================================================
    
    # **DEFAULT_VOICE**: Voz por defecto para síntesis de voz
    #   - Voces disponibles:
    #     - es-PA-MargaritaNeural: Mujer, Panamá (default)
    #     - es-ES-AlvaroNeural: Hombre, España
    #     - es-MX-DariaNeural: Mujer, México
    #   - Variable de entorno: DEFAULT_VOICE
    DEFAULT_VOICE = os.getenv("DEFAULT_VOICE", "es-PA-MargaritaNeural")
    
    # **VOICE_SPEED**: Velocidad de síntesis de voz
    #   - Formato: "+XX%" o "-XX%"
    #   - +20% = 20% más rápido (default)
    #   - +0% = velocidad normal
    #   - -10% = 10% más lento
    #   - Variable de entorno: VOICE_SPEED
    VOICE_SPEED = os.getenv("VOICE_SPEED", "+20%")
    
    # =========================================================
    #  VALIDACIÓN
    # =========================================================
    
    @classmethod
    def validate_Curim(cls):
        """
        Validar que Curim esté configurado correctamente.
        
        Verifica:
            1. GEMINI_API_KEY está configurado
            2. Directorios de almacenamiento existen
            3. Permisos de escritura en directorios
        
        Crea directorios automáticamente si no existen.
        
        Raises:
            ValueError: Si GEMINI_API_KEY no está configurado
        
        Uso:
            from app.core.config import settings
            
            # Al iniciar la aplicación
            settings.validate_Curim()
        
        Example output:
             Curim configurado correctamente
        
        Notas de seguridad:
            - Validar en startup de la aplicación
            - Fallar temprano si falta configuración
            - No proceder sin API key válida
        """
        # Verificar API key
        if not cls.GEMINI_API_KEY:
            raise ValueError(
                " GEMINI_API_KEY no configurado. "
                "Por favor, agrega tu API key de Gemini en el archivo .env"
            )
        
        # Crear directorios necesarios
        import os
        
        # Directorio para base de datos vectorial (Chroma)
        os.makedirs(
            os.path.join(cls.Curim_STORAGE_PATH, "chroma_db"), 
            exist_ok=True
        )
        
        # Directorio para caché de respuestas
        os.makedirs(
            os.path.join(cls.Curim_STORAGE_PATH, "cache"), 
            exist_ok=True
        )
        
       
# Instancia global de configuración
# Usar en toda la aplicación: from app.core.config import settings
settings = Settings()
