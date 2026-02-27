"""
Capa de almacenamiento de estado de sesión.
Soporta modo en memoria (desarrollo) y Redis (producción).
"""

import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Optional
import asyncio

logger = logging.getLogger(__name__)


class SessionStore(ABC):
    """Interfaz para el almacén de estado de sesión."""

    @abstractmethod
    async def set(self, key: str, value: Any, ttl_seconds: int = 3600) -> None:
        ...

    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        ...

    @abstractmethod
    async def delete(self, key: str) -> None:
        ...

    @abstractmethod
    async def increment(self, key: str) -> int:
        ...


class InMemorySessionStore(SessionStore):
    """
    Implementación en memoria para desarrollo single-worker.
    NO usar en producción multi-worker.
    """

    def __init__(self):
        self._store: dict[str, Any] = {}
        self._lock = asyncio.Lock()

    async def set(self, key: str, value: Any, ttl_seconds: int = 3600) -> None:
        async with self._lock:
            self._store[key] = value

    async def get(self, key: str) -> Optional[Any]:
        async with self._lock:
            return self._store.get(key)

    async def delete(self, key: str) -> None:
        async with self._lock:
            self._store.pop(key, None)

    async def increment(self, key: str) -> int:
        async with self._lock:
            val = self._store.get(key, 0) + 1
            self._store[key] = val
            return val


class RedisSessionStore(SessionStore):
    """
    Implementación Redis para producción multi-worker.
    Requiere: pip install redis[asyncio]
    """

    def __init__(self, redis_url: str):
        import redis.asyncio as aioredis
        self._redis = aioredis.from_url(
            redis_url,
            encoding="utf-8",
            decode_responses=True,
        )

    async def set(self, key: str, value: Any, ttl_seconds: int = 3600) -> None:
        serialized = json.dumps(value)
        await self._redis.setex(key, ttl_seconds, serialized)

    async def get(self, key: str) -> Optional[Any]:
        raw = await self._redis.get(key)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.warning(f"SessionStore: invalid JSON for key {key}")
            return None

    async def delete(self, key: str) -> None:
        await self._redis.delete(key)

    async def increment(self, key: str) -> int:
        return await self._redis.incr(key)

    async def close(self) -> None:
        await self._redis.aclose()


def create_session_store() -> SessionStore:
    """
    Factory que elige la implementación según el entorno.
    Configurar REDIS_URL en producción.
    """
    redis_url = os.getenv("REDIS_URL")

    if redis_url:
        logger.info(f"Usando RedisSessionStore: {redis_url}")
        return RedisSessionStore(redis_url)

    logger.warning(
        "REDIS_URL no configurada. Usando InMemorySessionStore. "
        "NO apto para múltiples workers."
    )
    return InMemorySessionStore()


# Instancia singleton — creada en startup
_session_store: Optional[SessionStore] = None


def get_session_store() -> SessionStore:
    if _session_store is None:
        raise RuntimeError(
            "SessionStore no inicializado. "
            "Llama a init_session_store() en el lifespan del app."
        )
    return _session_store


def init_session_store() -> SessionStore:
    global _session_store
    _session_store = create_session_store()
    return _session_store