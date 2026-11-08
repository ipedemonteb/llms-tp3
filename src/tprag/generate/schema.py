"""Salida estructurada del LLM (generador.md §2.5): lista de reemplazos + una decisión por import problemático.

Las claves van sin acentos (`despues`) para no depender de cómo el proveedor maneja el esquema.
"""

from typing import Literal

from pydantic import BaseModel, Field

REPLACED = "reemplazado"
NO_REPLACEMENT = "sin_reemplazo_confiable"


class Edit(BaseModel):
    antes: str = Field(description="Fragmento copiado exacto del script, con su sangría; tiene que aparecer una sola vez")
    despues: str = Field(description="Texto que reemplaza al fragmento")


class ImportDecision(BaseModel):
    """El orden de los campos importa: el modelo los genera en este orden, así que describe qué hace el script y qué
    fragmento hace lo mismo antes de decidir (generador.md §4.4)."""

    module: str = Field(description="El módulo problemático, tal como figura en la lista recibida")
    script_usage: str = Field(description="Qué hace el script con este import, en una oración")
    matching_fragment: str | None = Field(
        description="Encabezado del fragmento que hace eso mismo (no algo parecido o del mismo tema); null si ninguno")
    decision: Literal["reemplazado", "sin_reemplazo_confiable"]
    replacement_package: str | None = Field(
        description="Paquete de reemplazo tal como aparece al comienzo del encabezado del fragmento; null si no hay")
    reason: str = Field(description="Motivo en castellano, una o dos oraciones")
    sources: list[str] = Field(description="Encabezados de los fragmentos usados")


class GeneratorOutput(BaseModel):
    imports: list[ImportDecision]
    edits: list[Edit]
