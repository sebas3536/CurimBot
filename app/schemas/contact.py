# app/schemas/contact.py
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel

class ContactBase(BaseModel):
    nombre: str
    email: str
    telefono: Optional[str] = None
    empresa: Optional[str] = None
    etiquetas: List[str] = []

class ContactCreate(ContactBase):
    pass

class ContactUpdate(ContactBase):
    pass

class ContactResponse(ContactBase):
    id: int
    fecha_creacion: datetime

    class Config:
        orm_mode = True

    @classmethod
    def from_orm_model(cls, model):
        return cls(
            id=model.id,
            nombre=model.nombre,
            email=model.email,
            telefono=model.telefono,
            empresa=model.empresa,
            etiquetas=model.etiquetas.split(",") if model.etiquetas else [],
            fecha_creacion=model.fecha_creacion
        )