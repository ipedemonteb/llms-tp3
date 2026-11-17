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


class ScriptOutput(BaseModel):
    """Respuesta con el script completo (formato "script", solo en el reintento de la iteración, iteracion_ej4.md
    §3.2): el mismo reporte por import, y el script corregido en vez de los reemplazos."""

    imports: list[ImportDecision]
    script: str = Field(description="El script completo corregido, de la primera a la última línea")


# Primer prompt v2 (iteracion_ej4.md, `__iter5`): antes de elegir el fragmento, el modelo lista todos los usos del import
# y, por cada paquete de los fragmentos, qué usos cubre. Clases aparte: las de arriba (v1) no cambian, ni su esquema ni
# la clave del caché de sus respuestas.


class Candidate(BaseModel):
    package: str = Field(description="Paquete tal como aparece al comienzo del encabezado de sus fragmentos")
    covered_usages: str = Field(description="Qué usos del script cubre este paquete según sus fragmentos")
    api_names: list[str] = Field(description="Nombres de su API que aparecen en los fragmentos y cubren esos usos")


class ImportDecisionV2(BaseModel):
    """Como `ImportDecision`, con `candidates` entre `script_usage` y `matching_fragment` (el orden importa: el modelo
    compara los paquetes antes de elegir el fragmento y de decidir)."""

    module: str = Field(description="El módulo problemático, tal como figura en la lista recibida")
    script_usage: str = Field(description="Todos los usos del import en el script: qué nombres usa y para qué")
    candidates: list[Candidate] = Field(description="Cada paquete de los encabezados de los fragmentos y qué usos cubre")
    matching_fragment: str | None = Field(
        description="Encabezado de un fragmento del paquete que cubre más usos del script; null si ninguno cubre "
                    "ninguno")
    decision: Literal["reemplazado", "sin_reemplazo_confiable"]
    replacement_package: str | None = Field(
        description="Paquete de reemplazo tal como aparece al comienzo del encabezado del fragmento; null si no hay")
    reason: str = Field(description="Motivo en castellano, una o dos oraciones")
    sources: list[str] = Field(description="Encabezados de los fragmentos usados")


class GeneratorOutputV2(GeneratorOutput):
    imports: list[ImportDecisionV2]


class ScriptOutputV2(ScriptOutput):
    imports: list[ImportDecisionV2]
