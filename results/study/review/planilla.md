# Revisión manual de la generación

> **DEPRECADA (2026-11-17): no se usa.** Las métricas automáticas se validan contra Claude Opus 5.5 con rúbrica común,
> a ciegas y con scripts ancla (`docs/agents/metricas.md` §3.8): `results/study/judge_rubric*.md` (Éxito automático y LLM as a Judge) y
> `results/study/faith_vs_opus.md` (Faithfulness). Se conserva solo por trazabilidad.

20 scripts de dev corregidos por gpt-4o-mini (metricas.md §3.8). Para cada uno, completá después de los dos puntos (sin borrar la etiqueta):

- `preserva:` `si`, `parcial` o `no`.
- `que_se_perdio:` qué se perdió o cambió, con el número de línea; vacío si nada.
- `errores_heuristica:` los números de la tabla en los que la heurística se equivoca, separados por coma, o `ninguno`.

Las instrucciones que recibe el juez para la misma pregunta (usá el mismo criterio):

> Vas a comparar dos versiones de un script de Python. En la versión modificada se reemplazó un import (el módulo indicado) por otra librería y se adaptó el código que lo usaba. No evalúes si la librería elegida es buena ni el estilo: evaluá si el script modificado sigue haciendo la misma tarea que el original. Fijate si alguna funcionalidad se perdió, cambió de comportamiento o quedó sin implementar (por ejemplo, código borrado, `pass`, `TODO`, valores fijos en lugar de cálculos). Respondé `si` si hace lo mismo, `parcial` si hace lo principal pero pierde o cambia algo, y `no` si ya no hace la tarea.

Los scripts están en orden al azar y la planilla no muestra qué corrida es cada uno ni qué dijo el juez.

## R01

- **Tarea del script:** Generar muestras cuasi-aleatorias de dos dimensiones con una secuencia de Halton, pidiendo dos tandas consecutivas.
- **Import reemplazado:** `wrenseq`

### Cambios

````diff
--- original
+++ modificado
@@ -3,5 +3,5 @@
 import sys
 
-import wrenseq
+from scipy.stats import qmc
 
 logger = logging.getLogger("halton_draws")
@@ -10,5 +10,5 @@
 def make_engine(dimensions, seed=None):
     # Crear el generador de secuencia de Halton con la cantidad de dimensiones pedida.
-    return wrenseq.Halton(dims=dimensions, seed=seed)
+    return qmc.Halton(d=dimensions)
 
 
@@ -17,5 +17,5 @@
     results = []
     for _ in range(batches):
-        results.append(engine.next_points(batch_size))
+        results.append(engine.random(batch_size))
     return results
````

<details><summary>Script original completo</summary>

````python
import json
import logging
import sys

import wrenseq

logger = logging.getLogger("halton_draws")


def make_engine(dimensions, seed=None):
    # Crear el generador de secuencia de Halton con la cantidad de dimensiones pedida.
    return wrenseq.Halton(dims=dimensions, seed=seed)


def draw_batches(engine, batch_size, batches):
    # Pedir varias tandas consecutivas: la secuencia continúa entre llamadas.
    results = []
    for _ in range(batches):
        results.append(engine.next_points(batch_size))
    return results


def main(argv):
    logging.basicConfig(level=logging.INFO)
    dims = int(argv[1]) if len(argv) > 1 else 2
    engine = make_engine(dims)
    for number, batch in enumerate(draw_batches(engine, 5, 2), start=1):
        logger.info("batch %d: %s", number, json.dumps([list(p) for p in batch]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
````

</details>

<details><summary>Script modificado completo</summary>

````python
import json
import logging
import sys

from scipy.stats import qmc

logger = logging.getLogger("halton_draws")


def make_engine(dimensions, seed=None):
    # Crear el generador de secuencia de Halton con la cantidad de dimensiones pedida.
    return qmc.Halton(d=dimensions)


def draw_batches(engine, batch_size, batches):
    # Pedir varias tandas consecutivas: la secuencia continúa entre llamadas.
    results = []
    for _ in range(batches):
        results.append(engine.random(batch_size))
    return results


def main(argv):
    logging.basicConfig(level=logging.INFO)
    dims = int(argv[1]) if len(argv) > 1 else 2
    engine = make_engine(dims)
    for number, batch in enumerate(draw_batches(engine, 5, 2), start=1):
        logger.info("batch %d: %s", number, json.dumps([list(p) for p in batch]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
````

</details>

### Preservación de la tarea

¿El script modificado hace la misma tarea que el original? `si` / `parcial` / `no`.

- preserva: parcial
- que_se_perdio: no se pasa mas el parametro seed

### Faithfulness (heurística)

Identificadores de la API nueva en las líneas cambiadas. La heurística dice que un identificador es fiel si aparece en los chunks que recibió el LLM. Marcá los números en los que **la heurística se equivoca** (p. ej. aparece en el contexto pero no como esa API, o no aparece y sí está).

| # | Identificador | Línea | Tipo | ¿En el contexto? (heurística) | Dónde aparece |
|---|---|---|---|---|---|
| 1 | `scipy.stats` | 5 | module | sí | `Several QMC samplers/engines are implemented. Here we look at two of the most used QMC methods: 'scipy.stats.qmc.Sobol' and 'scipy.stats.qmc` |
| 2 | `qmc` | 5 | import | sí | `Several QMC samplers/engines are implemented. Here we look at two of the most used QMC methods: 'scipy.stats.qmc.Sobol' and 'scipy.stats.qmc` |
| 3 | `d` | 12 | kwarg | sí | `engine = qmc.Halton(d=2)` |
| 4 | `Halton` | 12 | attr | sí | `Several QMC samplers/engines are implemented. Here we look at two of the most used QMC methods: 'scipy.stats.qmc.Sobol' and 'scipy.stats.qmc` |

- errores_heuristica: ninguno

## R02

- **Tarea del script:** middleware que valida un JWT y sus claims (vencimiento, emisor, audiencia)
- **Import reemplazado:** `jose`

### Cambios

````diff
--- original
+++ modificado
@@ -8,5 +8,6 @@
 import os
 
-from jose import ExpiredSignatureError, JWTError, jwt
+from authlib.jose import jwt
+from authlib.jose import ExpiredSignatureError, JWTError
 
 log = logging.getLogger("auth")
@@ -49,5 +50,12 @@
 
         try:
-            claims = validar_token(token)
+            claims = jwt.decode(
+        token,
+        SECRETO,
+        algorithms=['HS256'],
+        audience=AUDIENCIA,
+        issuer=EMISOR,
+        options={'require_exp': True, 'require_sub': True, 'leeway': 30},
+    )
         except ExpiredSignatureError:
             return _rechazar(start_response, "401 Unauthorized", "token vencido")
````

<details><summary>Script original completo</summary>

````python
"""Middleware WSGI que valida el token Bearer de cada pedido antes de llegar a la aplicación.

Rechaza tokens vencidos, de otro emisor o para otra audiencia, y deja los claims en environ["auth.claims"].
"""

import json
import logging
import os

from jose import ExpiredSignatureError, JWTError, jwt

log = logging.getLogger("auth")

SECRETO = os.environ["JWT_SECRET"]
EMISOR = os.environ.get("JWT_ISSUER", "https://auth.internal.example")
AUDIENCIA = os.environ.get("JWT_AUDIENCE", "api-gateway")
RUTAS_PUBLICAS = {"/health", "/metrics"}


def _rechazar(start_response, estado, motivo):
    cuerpo = json.dumps({"error": motivo}).encode()
    start_response(estado, [("Content-Type", "application/json"), ("Content-Length", str(len(cuerpo)))])
    return [cuerpo]


def validar_token(token):
    return jwt.decode(
        token,
        SECRETO,
        algorithms=["HS256"],
        audience=AUDIENCIA,
        issuer=EMISOR,
        options={"require_exp": True, "require_sub": True, "leeway": 30},
    )


class ValidarJWT:
    def __init__(self, app):
        self.app = app

    def __call__(self, environ, start_response):
        if environ.get("PATH_INFO") in RUTAS_PUBLICAS:
            return self.app(environ, start_response)

        encabezado = environ.get("HTTP_AUTHORIZATION", "")
        esquema, _, token = encabezado.partition(" ")
        if esquema.lower() != "bearer" or not token:
            return _rechazar(start_response, "401 Unauthorized", "falta el token")

        try:
            claims = validar_token(token)
        except ExpiredSignatureError:
            return _rechazar(start_response, "401 Unauthorized", "token vencido")
        except JWTError as exc:
            log.info("token rechazado: %s", exc)
            return _rechazar(start_response, "401 Unauthorized", "token inválido")

        environ["auth.claims"] = claims
        environ["auth.subject"] = claims["sub"]
        return self.app(environ, start_response)
````

</details>

<details><summary>Script modificado completo</summary>

````python
"""Middleware WSGI que valida el token Bearer de cada pedido antes de llegar a la aplicación.

Rechaza tokens vencidos, de otro emisor o para otra audiencia, y deja los claims en environ["auth.claims"].
"""

import json
import logging
import os

from authlib.jose import jwt
from authlib.jose import ExpiredSignatureError, JWTError

log = logging.getLogger("auth")

SECRETO = os.environ["JWT_SECRET"]
EMISOR = os.environ.get("JWT_ISSUER", "https://auth.internal.example")
AUDIENCIA = os.environ.get("JWT_AUDIENCE", "api-gateway")
RUTAS_PUBLICAS = {"/health", "/metrics"}


def _rechazar(start_response, estado, motivo):
    cuerpo = json.dumps({"error": motivo}).encode()
    start_response(estado, [("Content-Type", "application/json"), ("Content-Length", str(len(cuerpo)))])
    return [cuerpo]


def validar_token(token):
    return jwt.decode(
        token,
        SECRETO,
        algorithms=["HS256"],
        audience=AUDIENCIA,
        issuer=EMISOR,
        options={"require_exp": True, "require_sub": True, "leeway": 30},
    )


class ValidarJWT:
    def __init__(self, app):
        self.app = app

    def __call__(self, environ, start_response):
        if environ.get("PATH_INFO") in RUTAS_PUBLICAS:
            return self.app(environ, start_response)

        encabezado = environ.get("HTTP_AUTHORIZATION", "")
        esquema, _, token = encabezado.partition(" ")
        if esquema.lower() != "bearer" or not token:
            return _rechazar(start_response, "401 Unauthorized", "falta el token")

        try:
            claims = jwt.decode(
        token,
        SECRETO,
        algorithms=['HS256'],
        audience=AUDIENCIA,
        issuer=EMISOR,
        options={'require_exp': True, 'require_sub': True, 'leeway': 30},
    )
        except ExpiredSignatureError:
            return _rechazar(start_response, "401 Unauthorized", "token vencido")
        except JWTError as exc:
            log.info("token rechazado: %s", exc)
            return _rechazar(start_response, "401 Unauthorized", "token inválido")

        environ["auth.claims"] = claims
        environ["auth.subject"] = claims["sub"]
        return self.app(environ, start_response)
````

</details>

### Preservación de la tarea

¿El script modificado hace la misma tarea que el original? `si` / `parcial` / `no`.

- preserva: no
- que_se_perdio: El middleware ya no puede validar tokens.

### Faithfulness (heurística)

Identificadores de la API nueva en las líneas cambiadas. La heurística dice que un identificador es fiel si aparece en los chunks que recibió el LLM. Marcá los números en los que **la heurística se equivoca** (p. ej. aparece en el contexto pero no como esa API, o no aparece y sí está).

| # | Identificador | Línea | Tipo | ¿En el contexto? (heurística) | Dónde aparece |
|---|---|---|---|---|---|
| 1 | `authlib.jose` | 10 | module | sí | `from authlib.jose import jwt` |
| 2 | `jwt` | 10 | import | sí | `from authlib.jose import jwt` |
| 3 | `ExpiredSignatureError` | 11 | import | no | — |
| 4 | `JWTError` | 11 | import | no | — |
| 5 | `algorithms` | 52 | kwarg | sí | `JSON Web Token (JWT) is structured by specs/rfc7515 or specs/rfc7516 with certain payload claims. The JWT implementation in Authlib has all ` |
| 6 | `audience` | 52 | kwarg | no | — |
| 7 | `issuer` | 52 | kwarg | no | — |
| 8 | `options` | 52 | kwarg | no | — |

- errores_heuristica: 5

## R03

- **Tarea del script:** Recorrer varias páginas de un catálogo de libros según un parámetro de cantidad de páginas y extraer título y precio.
- **Import reemplazado:** `cobaltrove`

### Cambios

````diff
--- original
+++ modificado
@@ -3,5 +3,5 @@
 from dataclasses import dataclass
 
-import cobaltrove
+import trafilatura
 
 logger = logging.getLogger("books")
@@ -14,5 +14,5 @@
 
 
-class BookCrawler(cobaltrove.ArgsCrawler[CrawlParams], cobaltrove.Crawler):
+class BookCrawler(trafilatura.spider.focused_crawler, trafilatura.spider.Crawler):
     name = "bookcrawler"
     start_urls = ["http://books.example.test/catalogue"]
@@ -22,5 +22,5 @@
         for base in self.start_urls:
             for index in range(1, self.args.pages + 1):
-                yield cobaltrove.Request(f"{base}/page-{index}.html")
+                yield trafilatura.Request(f"{base}/page-{index}.html")
 
     def parse(self, response):
@@ -42,6 +42,6 @@
         logger.error("pages must be >= 1")
         return 2
-    runner = cobaltrove.CrawlerRunner()
-    runner.crawl(BookCrawler, args=CrawlParams(pages=pages))
+    runner = trafilatura.CrawlerRunner()
+    runner.crawl(BookCrawler, args=pages)
     runner.start()
     return 0
````

<details><summary>Script original completo</summary>

````python
import logging
import sys
from dataclasses import dataclass

import cobaltrove

logger = logging.getLogger("books")


@dataclass
class CrawlParams:
    # Parámetros que recibe el crawler por línea de comandos.
    pages: int


class BookCrawler(cobaltrove.ArgsCrawler[CrawlParams], cobaltrove.Crawler):
    name = "bookcrawler"
    start_urls = ["http://books.example.test/catalogue"]

    async def start(self):
        # Generar una request por cada página del catálogo.
        for base in self.start_urls:
            for index in range(1, self.args.pages + 1):
                yield cobaltrove.Request(f"{base}/page-{index}.html")

    def parse(self, response):
        # Seguir cada link de libro encontrado en la página.
        for link in response.select("article.product_pod h3 a::attr(href)").getall():
            yield response.follow(link, self.parse_book)

    def parse_book(self, response):
        yield {
            "title": response.select("h1::text").get(),
            "price": response.select("p.price_color::text").get(),
        }


def main(argv):
    logging.basicConfig(level=logging.INFO)
    pages = int(argv[1]) if len(argv) > 1 else 1
    if pages < 1:
        logger.error("pages must be >= 1")
        return 2
    runner = cobaltrove.CrawlerRunner()
    runner.crawl(BookCrawler, args=CrawlParams(pages=pages))
    runner.start()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
````

</details>

<details><summary>Script modificado completo</summary>

````python
import logging
import sys
from dataclasses import dataclass

import trafilatura

logger = logging.getLogger("books")


@dataclass
class CrawlParams:
    # Parámetros que recibe el crawler por línea de comandos.
    pages: int


class BookCrawler(trafilatura.spider.focused_crawler, trafilatura.spider.Crawler):
    name = "bookcrawler"
    start_urls = ["http://books.example.test/catalogue"]

    async def start(self):
        # Generar una request por cada página del catálogo.
        for base in self.start_urls:
            for index in range(1, self.args.pages + 1):
                yield trafilatura.Request(f"{base}/page-{index}.html")

    def parse(self, response):
        # Seguir cada link de libro encontrado en la página.
        for link in response.select("article.product_pod h3 a::attr(href)").getall():
            yield response.follow(link, self.parse_book)

    def parse_book(self, response):
        yield {
            "title": response.select("h1::text").get(),
            "price": response.select("p.price_color::text").get(),
        }


def main(argv):
    logging.basicConfig(level=logging.INFO)
    pages = int(argv[1]) if len(argv) > 1 else 1
    if pages < 1:
        logger.error("pages must be >= 1")
        return 2
    runner = trafilatura.CrawlerRunner()
    runner.crawl(BookCrawler, args=pages)
    runner.start()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
````

</details>

### Preservación de la tarea

¿El script modificado hace la misma tarea que el original? `si` / `parcial` / `no`.

- preserva: no
- que_se_perdio: falla por imports y parametros inexistentes

### Faithfulness (heurística)

Identificadores de la API nueva en las líneas cambiadas. La heurística dice que un identificador es fiel si aparece en los chunks que recibió el LLM. Marcá los números en los que **la heurística se equivoca** (p. ej. aparece en el contexto pero no como esa API, o no aparece y sí está).

| # | Identificador | Línea | Tipo | ¿En el contexto? (heurística) | Dónde aparece |
|---|---|---|---|---|---|
| 1 | `trafilatura` | 5 | module | sí | `from trafilatura.spider import focused_crawler` |
| 2 | `focused_crawler` | 16 | attr | sí | `The 'focused_crawler()' function integrates all necessary components and can be customized using various arguments. To use it, you will need` |
| 3 | `Crawler` | 16 | attr | no | — |
| 4 | `spider` | 16 | attr | sí | `from trafilatura.spider import focused_crawler` |
| 5 | `args` | 45 | kwarg | no | — |
| 6 | `CrawlerRunner` | 44 | attr | no | — |
| 7 | `crawl` | 45 | attr | sí | `- 'todo': provide a previously generated list of pages to visit (i.e. a crawl frontier)` |
| 8 | `Request` | 24 | attr | sí | `Instead of implementing a 'scrapy.Spider.start' method that yields 'scrapy.Request' objects from URLs, you can define a 'scrapy.Spider.start` |

- errores_heuristica: 7, 8

## R04

- **Tarea del script:** Validate a list of user records (name and integer id, coercing string ids) against a typed dictionary schema, print the result, report a nested error, and serialize the output to JSON.
- **Import reemplazado:** `schemavault`

### Cambios

````diff
--- original
+++ modificado
@@ -4,5 +4,10 @@
 from typing import TypedDict
 
-import schemavault
+from marshmallow import Schema, fields
+
+class UserSchema(Schema):
+    name = fields.Str()
+    id = fields.Int()
+
 
 logger = logging.getLogger("schema_demo")
````

<details><summary>Script original completo</summary>

````python
import json
import logging
import sys
from typing import TypedDict

import schemavault

logger = logging.getLogger("schema_demo")


class User(TypedDict):
    name: str
    id: int


user_list_schema = schemavault.ListSchema(User)


def validate_users(raw_records):
    """Check the raw records against the list schema and return the coerced data."""
    return user_list_schema.check(raw_records)


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    users = validate_users([{"name": "Fred", "id": "3"}])
    print(repr(users))

    try:
        validate_users([{"name": "Fred", "id": "wrong", "other": "no"}])
    except schemavault.SchemaError as error:
        logger.warning("Validation failed: %s", error)
        print(error)

    payload = json.dumps(users, separators=(",", ":"), default=str)
    print(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
````

</details>

<details><summary>Script modificado completo</summary>

````python
import json
import logging
import sys
from typing import TypedDict

from marshmallow import Schema, fields

class UserSchema(Schema):
    name = fields.Str()
    id = fields.Int()


logger = logging.getLogger("schema_demo")


class User(TypedDict):
    name: str
    id: int


user_list_schema = schemavault.ListSchema(User)


def validate_users(raw_records):
    """Check the raw records against the list schema and return the coerced data."""
    return user_list_schema.check(raw_records)


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    users = validate_users([{"name": "Fred", "id": "3"}])
    print(repr(users))

    try:
        validate_users([{"name": "Fred", "id": "wrong", "other": "no"}])
    except schemavault.SchemaError as error:
        logger.warning("Validation failed: %s", error)
        print(error)

    payload = json.dumps(users, separators=(",", ":"), default=str)
    print(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
````

</details>

### Preservación de la tarea

¿El script modificado hace la misma tarea que el original? `si` / `parcial` / `no`.

- preserva: no
- que_se_perdio: falla por errores de imports y funciones incorrectas.

### Faithfulness (heurística)

Identificadores de la API nueva en las líneas cambiadas. La heurística dice que un identificador es fiel si aparece en los chunks que recibió el LLM. Marcá los números en los que **la heurística se equivoca** (p. ej. aparece en el contexto pero no como esa API, o no aparece y sí está).

| # | Identificador | Línea | Tipo | ¿En el contexto? (heurística) | Dónde aparece |
|---|---|---|---|---|---|
| 1 | `marshmallow` | 6 | module | sí | `from marshmallow import Schema, fields` |
| 2 | `Schema` | 6 | import | sí | `from marshmallow import Schema, fields` |
| 3 | `fields` | 6 | import | sí | `Create a schema by defining a class with variables mapping attribute names to 'fields.Field' objects.` |
| 4 | `Str` | 9 | attr | sí | `name = fields.Str()` |
| 5 | `Int` | 10 | attr | sí | `age = fields.Int(validate=validate.Range(min=18, max=40))` |

- errores_heuristica: ninguno

## R05

- **Tarea del script:** Extraer todas las imágenes embebidas de un PDF y guardarlas como PNG, convirtiendo las de CMYK a RGB.
- **Import reemplazado:** `imgharvestry`

### Cambios

````diff
--- original
+++ modificado
@@ -4,5 +4,5 @@
 from pathlib import Path
 
-import imgharvestry
+import pymupdf
 
 logger = logging.getLogger("extraer_imagenes")
@@ -12,15 +12,15 @@
     """Guarda cada imagen del PDF en out_dir y devuelve la cantidad guardada."""
     out_dir.mkdir(parents=True, exist_ok=True)
-    document = imgharvestry.open_document(str(pdf_path))
+    document = pymupdf.open(str(pdf_path))
     saved = 0
 
-    for page_number, page in enumerate(document.pages()):
-        for image_number, image_info in enumerate(page.list_images()):
-            xref = image_info[0]
-            pixels = imgharvestry.Bitmap.from_document(document, xref)
-            if pixels.channels > 4:  # las imágenes CMYK se pasan a RGB
-                pixels = imgharvestry.Bitmap.convert(pixels, imgharvestry.COLORSPACE_RGB)
+    for page_number, page in enumerate(document):
+        for image_index, img in enumerate(page.get_images()):
+            xref = img[0]
+            pixels = pymupdf.Pixmap(document, xref)
+            if pixels.n - pixels.alpha > 3: # CMYK: convert to RGB first
+                pixels = pymupdf.Pixmap(pymupdf.csRGB, pixels)
             target = out_dir / f"pagina{page_number}_imagen{image_number}.png"
-            pixels.write_png(target)
+            pixels.save(target)
             saved += 1
             logger.debug("Imagen guardada: %s", target)
````

<details><summary>Script original completo</summary>

````python
import argparse
import logging
import sys
from pathlib import Path

import imgharvestry

logger = logging.getLogger("extraer_imagenes")


def extract_images(pdf_path: Path, out_dir: Path) -> int:
    """Guarda cada imagen del PDF en out_dir y devuelve la cantidad guardada."""
    out_dir.mkdir(parents=True, exist_ok=True)
    document = imgharvestry.open_document(str(pdf_path))
    saved = 0

    for page_number, page in enumerate(document.pages()):
        for image_number, image_info in enumerate(page.list_images()):
            xref = image_info[0]
            pixels = imgharvestry.Bitmap.from_document(document, xref)
            if pixels.channels > 4:  # las imágenes CMYK se pasan a RGB
                pixels = imgharvestry.Bitmap.convert(pixels, imgharvestry.COLORSPACE_RGB)
            target = out_dir / f"pagina{page_number}_imagen{image_number}.png"
            pixels.write_png(target)
            saved += 1
            logger.debug("Imagen guardada: %s", target)
    return saved


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Extrae las imágenes de un PDF.")
    parser.add_argument("pdf", type=Path, help="Ruta del PDF de entrada")
    parser.add_argument("--out", type=Path, default=Path("imagenes"), help="Carpeta de salida")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO)
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        total = extract_images(args.pdf, args.out)
    except OSError as exc:
        logger.error("Error al leer el PDF: %s", exc)
        return 1
    logger.info("Se guardaron %d imágenes en %s", total, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
````

</details>

<details><summary>Script modificado completo</summary>

````python
import argparse
import logging
import sys
from pathlib import Path

import pymupdf

logger = logging.getLogger("extraer_imagenes")


def extract_images(pdf_path: Path, out_dir: Path) -> int:
    """Guarda cada imagen del PDF en out_dir y devuelve la cantidad guardada."""
    out_dir.mkdir(parents=True, exist_ok=True)
    document = pymupdf.open(str(pdf_path))
    saved = 0

    for page_number, page in enumerate(document):
        for image_index, img in enumerate(page.get_images()):
            xref = img[0]
            pixels = pymupdf.Pixmap(document, xref)
            if pixels.n - pixels.alpha > 3: # CMYK: convert to RGB first
                pixels = pymupdf.Pixmap(pymupdf.csRGB, pixels)
            target = out_dir / f"pagina{page_number}_imagen{image_number}.png"
            pixels.save(target)
            saved += 1
            logger.debug("Imagen guardada: %s", target)
    return saved


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Extrae las imágenes de un PDF.")
    parser.add_argument("pdf", type=Path, help="Ruta del PDF de entrada")
    parser.add_argument("--out", type=Path, default=Path("imagenes"), help="Carpeta de salida")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO)
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        total = extract_images(args.pdf, args.out)
    except OSError as exc:
        logger.error("Error al leer el PDF: %s", exc)
        return 1
    logger.info("Se guardaron %d imágenes en %s", total, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
````

</details>

### Preservación de la tarea

¿El script modificado hace la misma tarea que el original? `si` / `parcial` / `no`.

- preserva: no
- que_se_perdio: falla por NameError en la primera imagen

### Faithfulness (heurística)

Identificadores de la API nueva en las líneas cambiadas. La heurística dice que un identificador es fiel si aparece en los chunks que recibió el LLM. Marcá los números en los que **la heurística se equivoca** (p. ej. aparece en el contexto pero no como esa API, o no aparece y sí está).

| # | Identificador | Línea | Tipo | ¿En el contexto? (heurística) | Dónde aparece |
|---|---|---|---|---|---|
| 1 | `pymupdf` | 6 | module | sí | `doc = pymupdf.open("input.pdf")` |
| 2 | `Pixmap` | 20 | attr | sí | `pix = pymupdf.Pixmap(doc, xref) # create a Pixmap` |
| 3 | `save` | 24 | attr | sí | `In order to convert a document to images, you should iterate the document pages, convert to pixmap and save, e.g.:` |
| 4 | `n` | 21 | attr | sí | `if pix.n - pix.alpha > 3: # CMYK: convert to RGB first` |
| 5 | `alpha` | 21 | attr | sí | `if pix.n - pix.alpha > 3: # CMYK: convert to RGB first` |
| 6 | `csRGB` | 22 | attr | sí | `pix = pymupdf.Pixmap(pymupdf.csRGB, pix)` |

- errores_heuristica: ninguno

## R06

- **Tarea del script:** Parsear un documento XML usando la librería rápida si está instalada y caer a la biblioteca estándar si no.
- **Import reemplazado:** `thimblexml`

### Cambios

````diff
--- original
+++ modificado
@@ -6,6 +6,6 @@
 
 try:
-    import thimblexml as etree
-    BACKEND = "thimblexml"
+    from lxml import etree
+    BACKEND = "lxml.etree"
 except ImportError:
     etree = stdlib_etree
````

<details><summary>Script original completo</summary>

````python
import logging
import sys
import xml.etree.ElementTree as stdlib_etree

logger = logging.getLogger("xml_backend")

try:
    import thimblexml as etree
    BACKEND = "thimblexml"
except ImportError:
    etree = stdlib_etree
    BACKEND = "xml.etree.ElementTree"


def load_document(path):
    # Parse the file with whichever backend was found at import time.
    tree = etree.parse(path)
    return tree.getroot()


def count_items(root):
    return sum(1 for _ in root.iter("item"))


def main(argv):
    logging.basicConfig(level=logging.INFO)
    if len(argv) < 2:
        logger.error("usage: xml_backend.py <file.xml>")
        return 2
    logger.info("running with %s", BACKEND)
    root = load_document(argv[1])
    logger.info("items found: %d", count_items(root))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
````

</details>

<details><summary>Script modificado completo</summary>

````python
import logging
import sys
import xml.etree.ElementTree as stdlib_etree

logger = logging.getLogger("xml_backend")

try:
    from lxml import etree
    BACKEND = "lxml.etree"
except ImportError:
    etree = stdlib_etree
    BACKEND = "xml.etree.ElementTree"


def load_document(path):
    # Parse the file with whichever backend was found at import time.
    tree = etree.parse(path)
    return tree.getroot()


def count_items(root):
    return sum(1 for _ in root.iter("item"))


def main(argv):
    logging.basicConfig(level=logging.INFO)
    if len(argv) < 2:
        logger.error("usage: xml_backend.py <file.xml>")
        return 2
    logger.info("running with %s", BACKEND)
    root = load_document(argv[1])
    logger.info("items found: %d", count_items(root))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
````

</details>

### Preservación de la tarea

¿El script modificado hace la misma tarea que el original? `si` / `parcial` / `no`.

- preserva: si
- que_se_perdio:

### Faithfulness (heurística)

Identificadores de la API nueva en las líneas cambiadas. La heurística dice que un identificador es fiel si aparece en los chunks que recibió el LLM. Marcá los números en los que **la heurística se equivoca** (p. ej. aparece en el contexto pero no como esa API, o no aparece y sí está).

| # | Identificador | Línea | Tipo | ¿En el contexto? (heurística) | Dónde aparece |
|---|---|---|---|---|---|
| 1 | `lxml` | 8 | module | sí | `prettyprint(tree)  # lxml 1.3.4 and later` |
| 2 | `etree` | 8 | import | sí | `tree = etree.parse(some_file_or_file_like_object)` |

- errores_heuristica: ninguno

## R07

- **Tarea del script:** Crear una herramienta de línea de comandos que recibe una ruta de configuración opcional y muestra su contenido si es un archivo, o un aviso si es un directorio o no existe.
- **Import reemplazado:** `optionweaver`

### Cambios

````diff
--- original
+++ modificado
@@ -4,16 +4,16 @@
 from typing import Optional
 
-import optionweaver
+import typer
 
 logger = logging.getLogger("config_cli")
-cli = optionweaver.Cli()
+cli = typer.Typer()
 
 
 @cli.command()
-def show_config(config: Optional[Path] = optionweaver.Opt(None)):
+def show_config(config: Optional[Path] = None):
     """Muestra la configuración indicada por la ruta."""
     if config is None:
         print("No hay archivo de configuración")
-        raise optionweaver.Halt()
+        raise typer.Abort()
     logger.info("Leyendo la ruta %s", config)
     if config.is_file():
````

<details><summary>Script original completo</summary>

````python
import logging
import sys
from pathlib import Path
from typing import Optional

import optionweaver

logger = logging.getLogger("config_cli")
cli = optionweaver.Cli()


@cli.command()
def show_config(config: Optional[Path] = optionweaver.Opt(None)):
    """Muestra la configuración indicada por la ruta."""
    if config is None:
        print("No hay archivo de configuración")
        raise optionweaver.Halt()
    logger.info("Leyendo la ruta %s", config)
    if config.is_file():
        text = config.read_text(encoding="utf-8")
        print(f"Contenido del archivo de configuración: {text}")
    elif config.is_dir():
        print("La ruta es un directorio, se usarán todos sus archivos de configuración")
    elif not config.exists():
        print("La configuración no existe")


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    cli()
    return 0


if __name__ == "__main__":
    sys.exit(main())
````

</details>

<details><summary>Script modificado completo</summary>

````python
import logging
import sys
from pathlib import Path
from typing import Optional

import typer

logger = logging.getLogger("config_cli")
cli = typer.Typer()


@cli.command()
def show_config(config: Optional[Path] = None):
    """Muestra la configuración indicada por la ruta."""
    if config is None:
        print("No hay archivo de configuración")
        raise typer.Abort()
    logger.info("Leyendo la ruta %s", config)
    if config.is_file():
        text = config.read_text(encoding="utf-8")
        print(f"Contenido del archivo de configuración: {text}")
    elif config.is_dir():
        print("La ruta es un directorio, se usarán todos sus archivos de configuración")
    elif not config.exists():
        print("La configuración no existe")


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    cli()
    return 0


if __name__ == "__main__":
    sys.exit(main())
````

</details>

### Preservación de la tarea

¿El script modificado hace la misma tarea que el original? `si` / `parcial` / `no`.

- preserva: parcial
- que_se_perdio: la opcion --config pasa a ser argumento posicional

### Faithfulness (heurística)

Identificadores de la API nueva en las líneas cambiadas. La heurística dice que un identificador es fiel si aparece en los chunks que recibió el LLM. Marcá los números en los que **la heurística se equivoca** (p. ej. aparece en el contexto pero no como esa API, o no aparece y sí está).

| # | Identificador | Línea | Tipo | ¿En el contexto? (heurística) | Dónde aparece |
|---|---|---|---|---|---|
| 1 | `typer` | 6 | module | sí | `import typer` |
| 2 | `Typer` | 9 | attr | sí | `You can tell Typer to not show the default value in the help text with 'show_default=False':` |
| 3 | `Abort` | 17 | attr | sí | `raise typer.Abort()` |

- errores_heuristica: ninguno

## R08

- **Tarea del script:** Generar continuaciones de texto para una lista de frases, usando el hardware disponible y lotes de tamaño fijo
- **Import reemplazado:** `ferrowave`

### Cambios

````diff
--- original
+++ modificado
@@ -3,5 +3,10 @@
 import sys
 
-from ferrowave import HardwareScan, make_engine
+from huggingface_hub import list_inference_endpoints_hardware
+
+
+# Usar acelerador si está disponible, si no CPU
+hw_list = list_inference_endpoints_hardware()
+chosen = next((hw for hw in hw_list if hw.accelerator == 'gpu' and hw.status == 'available'), None)
 
 logger = logging.getLogger(__name__)
````

<details><summary>Script original completo</summary>

````python
import json
import logging
import sys

from ferrowave import HardwareScan, make_engine

logger = logging.getLogger(__name__)

SEEDS = [
    "el secreto para hornear una torta buenísima es",
    "una tostada con palta es",
    "buenos aires tiene",
    "las empanadas son",
]


def choose_hardware():
    # Usar acelerador si está disponible, si no CPU
    scan = HardwareScan()
    chosen = scan.best_target()
    logger.info("Usando hardware: %s", chosen)
    return chosen


def build_engine(repo_id, target, lot_size):
    return make_engine(kind="continuacion", repo=repo_id, target=target, lot_size=lot_size)


def continue_seeds(engine, seeds):
    outputs = engine(seeds)
    # Cada salida es una lista con un diccionario por continuación
    return [item[0]["texto"] for item in outputs]


def main():
    logging.basicConfig(level=logging.INFO)
    target = choose_hardware()
    engine = build_engine("vanta/lumen-mini-1b", target, lot_size=2)
    texts = continue_seeds(engine, SEEDS)
    print(json.dumps(dict(zip(SEEDS, texts)), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
````

</details>

<details><summary>Script modificado completo</summary>

````python
import json
import logging
import sys

from huggingface_hub import list_inference_endpoints_hardware


# Usar acelerador si está disponible, si no CPU
hw_list = list_inference_endpoints_hardware()
chosen = next((hw for hw in hw_list if hw.accelerator == 'gpu' and hw.status == 'available'), None)

logger = logging.getLogger(__name__)

SEEDS = [
    "el secreto para hornear una torta buenísima es",
    "una tostada con palta es",
    "buenos aires tiene",
    "las empanadas son",
]


def choose_hardware():
    # Usar acelerador si está disponible, si no CPU
    scan = HardwareScan()
    chosen = scan.best_target()
    logger.info("Usando hardware: %s", chosen)
    return chosen


def build_engine(repo_id, target, lot_size):
    return make_engine(kind="continuacion", repo=repo_id, target=target, lot_size=lot_size)


def continue_seeds(engine, seeds):
    outputs = engine(seeds)
    # Cada salida es una lista con un diccionario por continuación
    return [item[0]["texto"] for item in outputs]


def main():
    logging.basicConfig(level=logging.INFO)
    target = choose_hardware()
    engine = build_engine("vanta/lumen-mini-1b", target, lot_size=2)
    texts = continue_seeds(engine, SEEDS)
    print(json.dumps(dict(zip(SEEDS, texts)), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
````

</details>

### Preservación de la tarea

¿El script modificado hace la misma tarea que el original? `si` / `parcial` / `no`.

- preserva: no
- que_se_perdio: agrega una consulta de hardware adicional en HuggingFace que no genera texto, y hay lineas que siguen usando imports que ya no estan importados.

### Faithfulness (heurística)

Identificadores de la API nueva en las líneas cambiadas. La heurística dice que un identificador es fiel si aparece en los chunks que recibió el LLM. Marcá los números en los que **la heurística se equivoca** (p. ej. aparece en el contexto pero no como esa API, o no aparece y sí está).

| # | Identificador | Línea | Tipo | ¿En el contexto? (heurística) | Dónde aparece |
|---|---|---|---|---|---|
| 1 | `huggingface_hub` | 5 | module | sí | `from huggingface_hub import list_inference_endpoints_hardware` |
| 2 | `list_inference_endpoints_hardware` | 5 | import | sí | `The 'accelerator', 'vendor', 'region', 'instance_type' and 'instance_size' values accepted by 'create_inference_endpoint' depend on each oth` |
| 3 | `accelerator` | 10 | attr | sí | `The 'accelerator', 'vendor', 'region', 'instance_type' and 'instance_size' values accepted by 'create_inference_endpoint' depend on each oth` |
| 4 | `status` | 10 | attr | sí | `if hw.accelerator == "gpu" and hw.status == "available":` |

- errores_heuristica: ninguno

## R09

- **Tarea del script:** Show two text panels in a terminal UI: one rendering markup with bold styling and a clickable action, and one showing raw markup without parsing it.
- **Import reemplazado:** `stackpanelkit`

### Cambios

````diff
--- original
+++ modificado
@@ -3,5 +3,6 @@
 import sys
 
-import stackpanelkit
+from textual.app import App, ComposeResult
+from textual.widgets import Static
 
 logger = logging.getLogger("content_demo")
@@ -31,10 +32,10 @@
 
 
-class ContentDemo(stackpanelkit.Session):
+class ContentDemo(App):
     style_sheet = STYLE_SHEET
 
-    def build(self):
-        yield stackpanelkit.Panel(MARKUP_TEXT, panel_id="markup_panel")
-        yield stackpanelkit.Panel(RAW_TEXT, panel_id="raw_panel", parse_markup=False)
+    def compose(self) -> ComposeResult:
+        yield Static(MARKUP_TEXT, id="markup_panel")
+        yield Static(RAW_TEXT, id="raw_panel", markup=False)
````

<details><summary>Script original completo</summary>

````python
import argparse
import logging
import sys

import stackpanelkit

logger = logging.getLogger("content_demo")

MARKUP_TEXT = """\
Hello, [bold $text on $primary]World[/]!

[@click=app.notify('Hello, World!')]Click me[/]
"""

RAW_TEXT = """\
Markup will [bold]not[/bold] be displayed.

Tags will be left in the output.

"""

STYLE_SHEET = """
Screen {
    Panel {
        height: 1fr;
    }
    #markup_panel { background: $primary-muted; }
    #raw_panel { background: $error-muted; }
}
"""


class ContentDemo(stackpanelkit.Session):
    style_sheet = STYLE_SHEET

    def build(self):
        yield stackpanelkit.Panel(MARKUP_TEXT, panel_id="markup_panel")
        yield stackpanelkit.Panel(RAW_TEXT, panel_id="raw_panel", parse_markup=False)


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Content markup demo")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    logging.basicConfig(level=args.log_level)
    logger.info("Starting content demo")
    ContentDemo().run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
````

</details>

<details><summary>Script modificado completo</summary>

````python
import argparse
import logging
import sys

from textual.app import App, ComposeResult
from textual.widgets import Static

logger = logging.getLogger("content_demo")

MARKUP_TEXT = """\
Hello, [bold $text on $primary]World[/]!

[@click=app.notify('Hello, World!')]Click me[/]
"""

RAW_TEXT = """\
Markup will [bold]not[/bold] be displayed.

Tags will be left in the output.

"""

STYLE_SHEET = """
Screen {
    Panel {
        height: 1fr;
    }
    #markup_panel { background: $primary-muted; }
    #raw_panel { background: $error-muted; }
}
"""


class ContentDemo(App):
    style_sheet = STYLE_SHEET

    def compose(self) -> ComposeResult:
        yield Static(MARKUP_TEXT, id="markup_panel")
        yield Static(RAW_TEXT, id="raw_panel", markup=False)


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Content markup demo")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    logging.basicConfig(level=args.log_level)
    logger.info("Starting content demo")
    ContentDemo().run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
````

</details>

### Preservación de la tarea

¿El script modificado hace la misma tarea que el original? `si` / `parcial` / `no`.

- preserva: parcial
- que_se_perdio: toma la hoja de estilos de CSS y no de stylesheet y se pierde el alto y los fondos de cada panel

### Faithfulness (heurística)

Identificadores de la API nueva en las líneas cambiadas. La heurística dice que un identificador es fiel si aparece en los chunks que recibió el LLM. Marcá los números en los que **la heurística se equivoca** (p. ej. aparece en el contexto pero no como esa API, o no aparece y sí está).

| # | Identificador | Línea | Tipo | ¿En el contexto? (heurística) | Dónde aparece |
|---|---|---|---|---|---|
| 1 | `textual.app` | 5 | module | sí | `from textual.app import App, ComposeResult` |
| 2 | `App` | 5 | import | sí | `from textual.app import App, ComposeResult` |
| 3 | `ComposeResult` | 5 | import | sí | `from textual.app import App, ComposeResult` |
| 4 | `textual.widgets` | 6 | module | sí | `from textual.widgets import Static` |
| 5 | `Static` | 6 | import | sí | `from textual.widgets import Static` |
| 6 | `markup` | 39 | kwarg | sí | `When building a custom widget you can embed color and style information in the string returned from the Widget's 'render()' method. Markup i` |

- errores_heuristica: ninguno

## R10

- **Tarea del script:** subir archivos a Google Drive autenticando con una cuenta de servicio
- **Import reemplazado:** `oauth2client.service_account`

### Cambios

````diff
--- original
+++ modificado
@@ -14,5 +14,5 @@
 from googleapiclient.discovery import build
 from googleapiclient.http import MediaFileUpload
-from oauth2client.service_account import ServiceAccountCredentials
+from google.oauth2 import service_account
 
 SCOPES = ["https://www.googleapis.com/auth/drive.file"]
@@ -24,5 +24,5 @@
 
 def servicio_drive():
-    credenciales = ServiceAccountCredentials.from_json_keyfile_name(os.environ["GOOGLE_SA_KEY"], SCOPES)
+    credenciales = service_account.Credentials.from_service_account_file(os.environ["GOOGLE_SA_KEY"], scopes=SCOPES)
     http = credenciales.authorize(httplib2.Http(timeout=60))
     return build("drive", "v3", http=http, cache_discovery=False)
````

<details><summary>Script original completo</summary>

````python
"""Respaldo nocturno: sube los reportes del día a una carpeta compartida de Google Drive.

Usa una cuenta de servicio (credenciales en GOOGLE_SA_KEY) con acceso de editor a la carpeta.
"""

import logging
import mimetypes
import os
import sys
from datetime import date
from pathlib import Path

import httplib2
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from oauth2client.service_account import ServiceAccountCredentials

SCOPES = ["https://www.googleapis.com/auth/drive.file"]
CARPETA_DRIVE = os.environ.get("DRIVE_FOLDER_ID", "")
DIR_REPORTES = Path(os.environ.get("REPORTES_DIR", "/var/reportes"))

log = logging.getLogger("respaldo")


def servicio_drive():
    credenciales = ServiceAccountCredentials.from_json_keyfile_name(os.environ["GOOGLE_SA_KEY"], SCOPES)
    http = credenciales.authorize(httplib2.Http(timeout=60))
    return build("drive", "v3", http=http, cache_discovery=False)


def subir(drive, ruta):
    tipo = mimetypes.guess_type(ruta.name)[0] or "application/octet-stream"
    metadata = {"name": ruta.name, "parents": [CARPETA_DRIVE]}
    media = MediaFileUpload(str(ruta), mimetype=tipo, resumable=ruta.stat().st_size > 5 * 1024 * 1024)
    archivo = drive.files().create(body=metadata, media_body=media, fields="id, size").execute()
    return archivo["id"]


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not CARPETA_DRIVE:
        log.error("falta DRIVE_FOLDER_ID")
        return 2

    hoy = date.today().strftime("%Y%m%d")
    reportes = sorted(DIR_REPORTES.glob(f"*{hoy}*"))
    if not reportes:
        log.info("no hay reportes de hoy")
        return 0

    drive = servicio_drive()
    errores = 0
    for ruta in reportes:
        try:
            log.info("subido %s (%s)", ruta.name, subir(drive, ruta))
        except Exception:
            log.exception("falló %s", ruta.name)
            errores += 1
    return 1 if errores else 0


if __name__ == "__main__":
    sys.exit(main())
````

</details>

<details><summary>Script modificado completo</summary>

````python
"""Respaldo nocturno: sube los reportes del día a una carpeta compartida de Google Drive.

Usa una cuenta de servicio (credenciales en GOOGLE_SA_KEY) con acceso de editor a la carpeta.
"""

import logging
import mimetypes
import os
import sys
from datetime import date
from pathlib import Path

import httplib2
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.oauth2 import service_account

SCOPES = ["https://www.googleapis.com/auth/drive.file"]
CARPETA_DRIVE = os.environ.get("DRIVE_FOLDER_ID", "")
DIR_REPORTES = Path(os.environ.get("REPORTES_DIR", "/var/reportes"))

log = logging.getLogger("respaldo")


def servicio_drive():
    credenciales = service_account.Credentials.from_service_account_file(os.environ["GOOGLE_SA_KEY"], scopes=SCOPES)
    http = credenciales.authorize(httplib2.Http(timeout=60))
    return build("drive", "v3", http=http, cache_discovery=False)


def subir(drive, ruta):
    tipo = mimetypes.guess_type(ruta.name)[0] or "application/octet-stream"
    metadata = {"name": ruta.name, "parents": [CARPETA_DRIVE]}
    media = MediaFileUpload(str(ruta), mimetype=tipo, resumable=ruta.stat().st_size > 5 * 1024 * 1024)
    archivo = drive.files().create(body=metadata, media_body=media, fields="id, size").execute()
    return archivo["id"]


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not CARPETA_DRIVE:
        log.error("falta DRIVE_FOLDER_ID")
        return 2

    hoy = date.today().strftime("%Y%m%d")
    reportes = sorted(DIR_REPORTES.glob(f"*{hoy}*"))
    if not reportes:
        log.info("no hay reportes de hoy")
        return 0

    drive = servicio_drive()
    errores = 0
    for ruta in reportes:
        try:
            log.info("subido %s (%s)", ruta.name, subir(drive, ruta))
        except Exception:
            log.exception("falló %s", ruta.name)
            errores += 1
    return 1 if errores else 0


if __name__ == "__main__":
    sys.exit(main())
````

</details>

### Preservación de la tarea

¿El script modificado hace la misma tarea que el original? `si` / `parcial` / `no`.

- preserva: no
- que_se_perdio: credenciales de google auth sin authorize. Error en uso de metodos. Falla al crear el servicio y no sube nada.

### Faithfulness (heurística)

Identificadores de la API nueva en las líneas cambiadas. La heurística dice que un identificador es fiel si aparece en los chunks que recibió el LLM. Marcá los números en los que **la heurística se equivoca** (p. ej. aparece en el contexto pero no como esa API, o no aparece y sí está).

| # | Identificador | Línea | Tipo | ¿En el contexto? (heurística) | Dónde aparece |
|---|---|---|---|---|---|
| 1 | `google.oauth2` | 16 | module | sí | `The google.oauth2.service_account.Credentials class is only used with OAuth 2.0 Service Accounts. No end-user is involved for these server-t` |
| 2 | `service_account` | 16 | import | sí | `The google.oauth2.service_account.Credentials class is only used with OAuth 2.0 Service Accounts. No end-user is involved for these server-t` |
| 3 | `scopes` | 26 | kwarg | sí | `scopes=['openid', 'https://www.googleapis.com/auth/userinfo.email', 'https://www.googleapis.com/auth/userinfo.profile'])` |
| 4 | `from_service_account_file` | 26 | attr | sí | `credentials = service_account.Credentials.from_service_account_file(` |
| 5 | `Credentials` | 26 | attr | sí | `The google.oauth2.service_account.Credentials class is only used with OAuth 2.0 Service Accounts. No end-user is involved for these server-t` |

- errores_heuristica: ninguno

## R11

- **Tarea del script:** tests generados a partir de una tabla de casos (tests generadores con yield)
- **Import reemplazado:** `nose.tools`

### Cambios

````diff
--- original
+++ modificado
@@ -3,5 +3,5 @@
 from datetime import datetime, timezone
 
-import nose.tools as nt
+import pendulum
 
 FORMATS = (
````

<details><summary>Script original completo</summary>

````python
"""Table-driven tests for the log timestamp normalizer (one generated test per case)."""

from datetime import datetime, timezone

import nose.tools as nt

FORMATS = (
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%d %H:%M:%S",
    "%d/%b/%Y:%H:%M:%S %z",  # apache access log
    "%b %d %H:%M:%S",  # syslog, no year
)


def normalize_timestamp(raw, default_year=2024):
    """Return an aware UTC datetime for the timestamp formats we see in the logs."""
    raw = raw.strip()
    for fmt in FORMATS:
        try:
            parsed = datetime.strptime(raw, fmt)
        except ValueError:
            continue
        if "%Y" not in fmt:
            parsed = parsed.replace(year=default_year)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    raise ValueError(f"unknown timestamp format: {raw!r}")


VALID = [
    ("2024-03-01T10:00:00+0000", datetime(2024, 3, 1, 10, 0, tzinfo=timezone.utc)),
    ("2024-03-01T07:00:00-0300", datetime(2024, 3, 1, 10, 0, tzinfo=timezone.utc)),
    ("2024-03-01 10:00:00", datetime(2024, 3, 1, 10, 0, tzinfo=timezone.utc)),
    ("01/Mar/2024:10:00:00 +0000", datetime(2024, 3, 1, 10, 0, tzinfo=timezone.utc)),
    ("Mar  1 10:00:00", datetime(2024, 3, 1, 10, 0, tzinfo=timezone.utc)),
]

INVALID = ["", "yesterday", "2024-13-01 10:00:00", "10:00"]


def check_valid(raw, expected):
    nt.eq_(normalize_timestamp(raw), expected)


def check_invalid(raw):
    nt.assert_raises(ValueError, normalize_timestamp, raw)


def test_valid_timestamps():
    for raw, expected in VALID:
        yield check_valid, raw, expected


def test_invalid_timestamps():
    for raw in INVALID:
        yield check_invalid, raw


def test_result_is_always_utc():
    for raw, _ in VALID:
        nt.assert_equal(normalize_timestamp(raw).utcoffset().total_seconds(), 0)
````

</details>

<details><summary>Script modificado completo</summary>

````python
"""Table-driven tests for the log timestamp normalizer (one generated test per case)."""

from datetime import datetime, timezone

import pendulum

FORMATS = (
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%d %H:%M:%S",
    "%d/%b/%Y:%H:%M:%S %z",  # apache access log
    "%b %d %H:%M:%S",  # syslog, no year
)


def normalize_timestamp(raw, default_year=2024):
    """Return an aware UTC datetime for the timestamp formats we see in the logs."""
    raw = raw.strip()
    for fmt in FORMATS:
        try:
            parsed = datetime.strptime(raw, fmt)
        except ValueError:
            continue
        if "%Y" not in fmt:
            parsed = parsed.replace(year=default_year)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    raise ValueError(f"unknown timestamp format: {raw!r}")


VALID = [
    ("2024-03-01T10:00:00+0000", datetime(2024, 3, 1, 10, 0, tzinfo=timezone.utc)),
    ("2024-03-01T07:00:00-0300", datetime(2024, 3, 1, 10, 0, tzinfo=timezone.utc)),
    ("2024-03-01 10:00:00", datetime(2024, 3, 1, 10, 0, tzinfo=timezone.utc)),
    ("01/Mar/2024:10:00:00 +0000", datetime(2024, 3, 1, 10, 0, tzinfo=timezone.utc)),
    ("Mar  1 10:00:00", datetime(2024, 3, 1, 10, 0, tzinfo=timezone.utc)),
]

INVALID = ["", "yesterday", "2024-13-01 10:00:00", "10:00"]


def check_valid(raw, expected):
    nt.eq_(normalize_timestamp(raw), expected)


def check_invalid(raw):
    nt.assert_raises(ValueError, normalize_timestamp, raw)


def test_valid_timestamps():
    for raw, expected in VALID:
        yield check_valid, raw, expected


def test_invalid_timestamps():
    for raw in INVALID:
        yield check_invalid, raw


def test_result_is_always_utc():
    for raw, _ in VALID:
        nt.assert_equal(normalize_timestamp(raw).utcoffset().total_seconds(), 0)
````

</details>

### Preservación de la tarea

¿El script modificado hace la misma tarea que el original? `si` / `parcial` / `no`.

- preserva: no
- que_se_perdio: cambio incorrecto de nose.tools por pendulum y hay lineas que siguen usando imports no importados.

### Faithfulness (heurística)

Identificadores de la API nueva en las líneas cambiadas. La heurística dice que un identificador es fiel si aparece en los chunks que recibió el LLM. Marcá los números en los que **la heurística se equivoca** (p. ej. aparece en el contexto pero no como esa API, o no aparece y sí está).

| # | Identificador | Línea | Tipo | ¿En el contexto? (heurística) | Dónde aparece |
|---|---|---|---|---|---|
| 1 | `pendulum` | 5 | module | sí | `import pendulum` |

- errores_heuristica: ninguno

## R12

- **Tarea del script:** refrescar y volver a guardar en disco los tokens OAuth de Google guardados por cuenta
- **Import reemplazado:** `oauth2client.file`

### Cambios

````diff
--- original
+++ modificado
@@ -10,5 +10,6 @@
 
 import httplib2
-from oauth2client.file import Storage
+from google_auth_oauthlib.flow import InstalledAppFlow
+from google.oauth2.credentials import Credentials
 
 TOKENS_DIR = Path("tokens")
@@ -17,6 +18,8 @@
 
 def refresh_account(path):
-    storage = Storage(str(path))
-    credentials = storage.get()
+    flow = InstalledAppFlow.from_client_secrets_file(
+        str(path),
+        scopes=['openid', 'https://www.googleapis.com/auth/userinfo.email', 'https://www.googleapis.com/auth/userinfo.profile'])
+    credentials = flow.run_local_server()
     if credentials is None:
         log.warning("%s: no credentials stored", path.stem)
@@ -26,5 +29,5 @@
         return False
     if credentials.access_token_expired or credentials.get_access_token().expires_in < 600:
-        credentials.refresh(httplib2.Http(timeout=30))
+        credentials.refresh(Request())
         storage.put(credentials)
         log.info("%s: refreshed", path.stem)
````

<details><summary>Script original completo</summary>

````python
"""Cron job: keep the stored Google OAuth tokens of every connected account fresh.

Each account has its credentials in tokens/<account>.json (written when the user connected the account). Refreshing
them ahead of time avoids failures in the nightly sync when a token expires mid-run.
"""

import logging
import sys
from pathlib import Path

import httplib2
from oauth2client.file import Storage

TOKENS_DIR = Path("tokens")
log = logging.getLogger("token-refresh")


def refresh_account(path):
    storage = Storage(str(path))
    credentials = storage.get()
    if credentials is None:
        log.warning("%s: no credentials stored", path.stem)
        return False
    if credentials.invalid:
        log.warning("%s: credentials revoked, user must reconnect", path.stem)
        return False
    if credentials.access_token_expired or credentials.get_access_token().expires_in < 600:
        credentials.refresh(httplib2.Http(timeout=30))
        storage.put(credentials)
        log.info("%s: refreshed", path.stem)
    return True


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    failures = [p.stem for p in sorted(TOKENS_DIR.glob("*.json")) if not refresh_account(p)]
    if failures:
        log.error("accounts needing attention: %s", ", ".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
````

</details>

<details><summary>Script modificado completo</summary>

````python
"""Cron job: keep the stored Google OAuth tokens of every connected account fresh.

Each account has its credentials in tokens/<account>.json (written when the user connected the account). Refreshing
them ahead of time avoids failures in the nightly sync when a token expires mid-run.
"""

import logging
import sys
from pathlib import Path

import httplib2
from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials

TOKENS_DIR = Path("tokens")
log = logging.getLogger("token-refresh")


def refresh_account(path):
    flow = InstalledAppFlow.from_client_secrets_file(
        str(path),
        scopes=['openid', 'https://www.googleapis.com/auth/userinfo.email', 'https://www.googleapis.com/auth/userinfo.profile'])
    credentials = flow.run_local_server()
    if credentials is None:
        log.warning("%s: no credentials stored", path.stem)
        return False
    if credentials.invalid:
        log.warning("%s: credentials revoked, user must reconnect", path.stem)
        return False
    if credentials.access_token_expired or credentials.get_access_token().expires_in < 600:
        credentials.refresh(Request())
        storage.put(credentials)
        log.info("%s: refreshed", path.stem)
    return True


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    failures = [p.stem for p in sorted(TOKENS_DIR.glob("*.json")) if not refresh_account(p)]
    if failures:
        log.error("accounts needing attention: %s", ", ".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
````

</details>

### Preservación de la tarea

¿El script modificado hace la misma tarea que el original? `si` / `parcial` / `no`.

- preserva: no
- que_se_perdio: abre flujo de autorizacion interactivo en el navegador en vez de leer y refrescar los tokens guardados

### Faithfulness (heurística)

Identificadores de la API nueva en las líneas cambiadas. La heurística dice que un identificador es fiel si aparece en los chunks que recibió el LLM. Marcá los números en los que **la heurística se equivoca** (p. ej. aparece en el contexto pero no como esa API, o no aparece y sí está).

| # | Identificador | Línea | Tipo | ¿En el contexto? (heurística) | Dónde aparece |
|---|---|---|---|---|---|
| 1 | `google_auth_oauthlib.flow` | 12 | module | sí | `from google_auth_oauthlib.flow import InstalledAppFlow` |
| 2 | `InstalledAppFlow` | 12 | import | sí | `from google_auth_oauthlib.flow import InstalledAppFlow` |
| 3 | `google.oauth2.credentials` | 13 | module | no | — |
| 4 | `Credentials` | 13 | import | sí | `The google.oauth2.service_account.Credentials class is only used with OAuth 2.0 Service Accounts. No end-user is involved for these server-t` |
| 5 | `scopes` | 20 | kwarg | sí | `scopes=['openid', 'https://www.googleapis.com/auth/userinfo.email', 'https://www.googleapis.com/auth/userinfo.profile'])` |
| 6 | `from_client_secrets_file` | 20 | attr | sí | `flow = InstalledAppFlow.from_client_secrets_file(` |
| 7 | `run_local_server` | 23 | attr | sí | `flow.run_local_server()` |
| 8 | `refresh` | 31 | attr | sí | `Already obtained access token? We can use 'OAuth2Auth' directly in requests. But this OAuth2Auth can not refresh token automatically for you` |

- errores_heuristica: 4, 8

## R13

- **Tarea del script:** Graficar la utilización de recursos como barras horizontales interrumpidas, con rangos de inicio y duración por recurso, y ordenar el eje Y de arriba hacia abajo.
- **Import reemplazado:** `gantterra`

### Cambios

````diff
--- original
+++ modificado
@@ -3,5 +3,10 @@
 import sys
 
-from gantterra import Chart
+from xlsxwriter import Workbook
+
+workbook = Workbook('chart.xlsx')
+worksheet = workbook.add_worksheet()
+
+chart = workbook.add_chart({'type': 'column'})
 
 logger = logging.getLogger(__name__)
@@ -28,11 +33,9 @@
 
 def draw(sources):
-    chart = Chart()
+    chart = workbook.add_chart({'type': 'column'})
     colors = {"disk": "tab:orange", "network": "tab:green"}
     for label, spans in sources.items():
-        chart.add_bars(spans, label, height=0.4, align="center", color=colors.get(label))
-    chart.set_xlim(0, 10)
-    chart.invert_y_axis()  # order the bars from top to bottom
-    chart.set_title("Resource usage")
+        chart.add_series({'values': spans})
+    worksheet.insert_chart('A7', chart)
     return chart
````

<details><summary>Script original completo</summary>

````python
import logging
import random
import sys

from gantterra import Chart

logger = logging.getLogger(__name__)


def build_sources(seed):
    # Cada recurso es una lista de (inicio, duración)
    rng = random.Random(seed)
    cpu_1 = [(0, 3), (3.5, 1), (5, 5)]
    cpu_2 = [(x, 0.5) for x in range(10)]
    cpu_3 = [(10 * rng.random(), 0.05) for _ in range(61)]
    cpu_4 = [(2, 1.7), (7, 1.2)]
    disk = [(1, 1.5)]
    network = [(10 * rng.random(), 0.05) for _ in range(10)]
    return {
        "CPU 1": cpu_1,
        "CPU 2": cpu_2,
        "CPU 3": cpu_3,
        "CPU 4": cpu_4,
        "disk": disk,
        "network": network,
    }


def draw(sources):
    chart = Chart()
    colors = {"disk": "tab:orange", "network": "tab:green"}
    for label, spans in sources.items():
        chart.add_bars(spans, label, height=0.4, align="center", color=colors.get(label))
    chart.set_xlim(0, 10)
    chart.invert_y_axis()  # order the bars from top to bottom
    chart.set_title("Resource usage")
    return chart


def main():
    logging.basicConfig(level=logging.INFO)
    sources = build_sources(seed=1)
    logger.info("Recursos: %s", ", ".join(sources))
    chart = draw(sources)
    chart.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
````

</details>

<details><summary>Script modificado completo</summary>

````python
import logging
import random
import sys

from xlsxwriter import Workbook

workbook = Workbook('chart.xlsx')
worksheet = workbook.add_worksheet()

chart = workbook.add_chart({'type': 'column'})

logger = logging.getLogger(__name__)


def build_sources(seed):
    # Cada recurso es una lista de (inicio, duración)
    rng = random.Random(seed)
    cpu_1 = [(0, 3), (3.5, 1), (5, 5)]
    cpu_2 = [(x, 0.5) for x in range(10)]
    cpu_3 = [(10 * rng.random(), 0.05) for _ in range(61)]
    cpu_4 = [(2, 1.7), (7, 1.2)]
    disk = [(1, 1.5)]
    network = [(10 * rng.random(), 0.05) for _ in range(10)]
    return {
        "CPU 1": cpu_1,
        "CPU 2": cpu_2,
        "CPU 3": cpu_3,
        "CPU 4": cpu_4,
        "disk": disk,
        "network": network,
    }


def draw(sources):
    chart = workbook.add_chart({'type': 'column'})
    colors = {"disk": "tab:orange", "network": "tab:green"}
    for label, spans in sources.items():
        chart.add_series({'values': spans})
    worksheet.insert_chart('A7', chart)
    return chart


def main():
    logging.basicConfig(level=logging.INFO)
    sources = build_sources(seed=1)
    logger.info("Recursos: %s", ", ".join(sources))
    chart = draw(sources)
    chart.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
````

</details>

### Preservación de la tarea

¿El script modificado hace la misma tarea que el original? `si` / `parcial` / `no`.

- preserva: no
- que_se_perdio: hace grafico de columnas en excel en vez del grafico de barras horizontales interrumpidas.

### Faithfulness (heurística)

Identificadores de la API nueva en las líneas cambiadas. La heurística dice que un identificador es fiel si aparece en los chunks que recibió el LLM. Marcá los números en los que **la heurística se equivoca** (p. ej. aparece en el contexto pero no como esa API, o no aparece y sí está).

| # | Identificador | Línea | Tipo | ¿En el contexto? (heurística) | Dónde aparece |
|---|---|---|---|---|---|
| 1 | `xlsxwriter` | 5 | module | sí | `import xlsxwriter` |
| 2 | `Workbook` | 5 | import | sí | `A chart object is created via the Workbook 'add_chart()' method where the chart type is specified:` |
| 3 | `add_worksheet` | 8 | attr | sí | `worksheet = workbook.add_worksheet()` |
| 4 | `add_chart` | 10 | attr | sí | `A chart object is created via the Workbook 'add_chart()' method where the chart type is specified:` |
| 5 | `insert_chart` | 39 | attr | sí | `It is then inserted into a worksheet as an embedded chart using the 'insert_chart' Worksheet method:` |
| 6 | `add_series` | 38 | attr | sí | `chart.add_series({'values': '=Sheet1!$A$1:$A$5'})` |

- errores_heuristica: ninguno

## R14

- **Tarea del script:** tests de una función de parseo de precios con asserts de nose.tools y @raises
- **Import reemplazado:** `nose.tools`

### Cambios

````diff
--- original
+++ modificado
@@ -7,5 +7,5 @@
 from decimal import Decimal, InvalidOperation
 
-from nose.tools import assert_equal, assert_in, assert_is_none, assert_true, raises
+from pytest import raises, mark
 
 _CURRENCY = {"$": "USD", "US$": "USD", "€": "EUR", "R$": "BRL"}
@@ -32,25 +32,25 @@
 
 def test_plain_number_uses_default_currency():
-    assert_equal(parse_price("10"), (Decimal("10.00"), "USD"))
+    assert parse_price("10") == (Decimal("10.00"), "USD")
 
 
 def test_symbol_is_mapped():
     amount, currency = parse_price("€ 3.5")
-    assert_equal(currency, "EUR")
-    assert_equal(amount, Decimal("3.50"))
+    assert currency == "EUR"
+    assert amount == Decimal("3.50")
 
 
 def test_thousands_separator():
-    assert_equal(parse_price("US$ 1,234.50")[0], Decimal("1234.50"))
+    assert parse_price("US$ 1,234.50")[0] == Decimal("1234.50")
 
 
 def test_known_currencies():
     for symbol in ("$", "US$", "€", "R$"):
-        assert_in(parse_price(symbol + "1")[1], {"USD", "EUR", "BRL"})
+        assert parse_price(symbol + "1")[1] in {"USD", "EUR", "BRL"}
 
 
 def test_blank_is_none():
-    assert_is_none(parse_price("   "))
-    assert_is_none(parse_price(""))
+    assert parse_price("   ") is None
+    assert parse_price("") is None
 
 
@@ -62,8 +62,10 @@
 @raises(ValueError)
 def test_garbage_raises():
-    parse_price("ten dollars")
+    with raises(ValueError):
+        parse_price("ten dollars")
 
 
 @raises(ValueError)
 def test_malformed_number_raises():
-    parse_price("1.2.3")
+    with raises(ValueError):
+        parse_price("1.2.3")
````

<details><summary>Script original completo</summary>

````python
"""Tests for the price parsing helpers used by the invoice importer.

Run with: nosetests -v test_prices.py
"""

import re
from decimal import Decimal, InvalidOperation

from nose.tools import assert_equal, assert_in, assert_is_none, assert_true, raises

_CURRENCY = {"$": "USD", "US$": "USD", "€": "EUR", "R$": "BRL"}
_AMOUNT = re.compile(r"^\s*(?P<cur>US\$|R\$|\$|€)?\s*(?P<num>[\d.,]+)\s*$")


def parse_price(text, default_currency="USD"):
    """'US$ 1,234.50' -> (Decimal('1234.50'), 'USD'). Empty input -> None."""
    if not text or not text.strip():
        return None
    match = _AMOUNT.match(text)
    if not match:
        raise ValueError(f"not a price: {text!r}")
    number = match.group("num").replace(",", "")
    try:
        amount = Decimal(number)
    except InvalidOperation:
        raise ValueError(f"bad amount: {text!r}")
    if amount < 0:
        raise ValueError("negative price")
    currency = _CURRENCY.get(match.group("cur"), default_currency)
    return amount.quantize(Decimal("0.01")), currency


def test_plain_number_uses_default_currency():
    assert_equal(parse_price("10"), (Decimal("10.00"), "USD"))


def test_symbol_is_mapped():
    amount, currency = parse_price("€ 3.5")
    assert_equal(currency, "EUR")
    assert_equal(amount, Decimal("3.50"))


def test_thousands_separator():
    assert_equal(parse_price("US$ 1,234.50")[0], Decimal("1234.50"))


def test_known_currencies():
    for symbol in ("$", "US$", "€", "R$"):
        assert_in(parse_price(symbol + "1")[1], {"USD", "EUR", "BRL"})


def test_blank_is_none():
    assert_is_none(parse_price("   "))
    assert_is_none(parse_price(""))


def test_rounding_is_two_decimals():
    amount, _ = parse_price("0.005")
    assert_true(amount.as_tuple().exponent == -2)


@raises(ValueError)
def test_garbage_raises():
    parse_price("ten dollars")


@raises(ValueError)
def test_malformed_number_raises():
    parse_price("1.2.3")
````

</details>

<details><summary>Script modificado completo</summary>

````python
"""Tests for the price parsing helpers used by the invoice importer.

Run with: nosetests -v test_prices.py
"""

import re
from decimal import Decimal, InvalidOperation

from pytest import raises, mark

_CURRENCY = {"$": "USD", "US$": "USD", "€": "EUR", "R$": "BRL"}
_AMOUNT = re.compile(r"^\s*(?P<cur>US\$|R\$|\$|€)?\s*(?P<num>[\d.,]+)\s*$")


def parse_price(text, default_currency="USD"):
    """'US$ 1,234.50' -> (Decimal('1234.50'), 'USD'). Empty input -> None."""
    if not text or not text.strip():
        return None
    match = _AMOUNT.match(text)
    if not match:
        raise ValueError(f"not a price: {text!r}")
    number = match.group("num").replace(",", "")
    try:
        amount = Decimal(number)
    except InvalidOperation:
        raise ValueError(f"bad amount: {text!r}")
    if amount < 0:
        raise ValueError("negative price")
    currency = _CURRENCY.get(match.group("cur"), default_currency)
    return amount.quantize(Decimal("0.01")), currency


def test_plain_number_uses_default_currency():
    assert parse_price("10") == (Decimal("10.00"), "USD")


def test_symbol_is_mapped():
    amount, currency = parse_price("€ 3.5")
    assert currency == "EUR"
    assert amount == Decimal("3.50")


def test_thousands_separator():
    assert parse_price("US$ 1,234.50")[0] == Decimal("1234.50")


def test_known_currencies():
    for symbol in ("$", "US$", "€", "R$"):
        assert parse_price(symbol + "1")[1] in {"USD", "EUR", "BRL"}


def test_blank_is_none():
    assert parse_price("   ") is None
    assert parse_price("") is None


def test_rounding_is_two_decimals():
    amount, _ = parse_price("0.005")
    assert_true(amount.as_tuple().exponent == -2)


@raises(ValueError)
def test_garbage_raises():
    with raises(ValueError):
        parse_price("ten dollars")


@raises(ValueError)
def test_malformed_number_raises():
    with raises(ValueError):
        parse_price("1.2.3")
````

</details>

### Preservación de la tarea

¿El script modificado hace la misma tarea que el original? `si` / `parcial` / `no`.

- preserva: no
- que_se_perdio: TypeError en importacion del modulo

### Faithfulness (heurística)

Identificadores de la API nueva en las líneas cambiadas. La heurística dice que un identificador es fiel si aparece en los chunks que recibió el LLM. Marcá los números en los que **la heurística se equivoca** (p. ej. aparece en el contexto pero no como esa API, o no aparece y sí está).

| # | Identificador | Línea | Tipo | ¿En el contexto? (heurística) | Dónde aparece |
|---|---|---|---|---|---|
| 1 | `pytest` | 9 | module | sí | `import pytest` |
| 2 | `raises` | 9 | import | sí | `with pytest.raises(ValueError, match=r".* 123 .*"):` |
| 3 | `mark` | 9 | import | no | — |

- errores_heuristica: ninguno

## R15

- **Tarea del script:** Ajustar un normalizador de media y varianza solo con los datos de referencia y aplicarlo tambien a los de evaluacion
- **Import reemplazado:** `meanvarkit`

### Cambios

````diff
--- original
+++ modificado
@@ -3,5 +3,5 @@
 import sys
 
-import meanvarkit
+from sklearn import preprocessing
 
 logger = logging.getLogger("normalize")
@@ -17,5 +17,5 @@
 
 def normalize_features(reference_rows, holdout_rows):
-    normalizer = meanvarkit.MeanVarNormalizer()
+    normalizer = preprocessing.Normalizer()
     # Solo los datos de referencia definen media y varianza, nunca los de evaluacion
     normalizer.learn(reference_rows)
@@ -29,6 +29,6 @@
     args = build_parser().parse_args(argv)
     try:
-        reference_rows = meanvarkit.read_table(args.reference)
-        holdout_rows = meanvarkit.read_table(args.holdout)
+        reference_rows = pd.read_csv(args.reference)
+        holdout_rows = pd.read_csv(args.holdout)
     except FileNotFoundError as exc:
         logger.error("Archivo no encontrado: %s", exc)
````

<details><summary>Script original completo</summary>

````python
import argparse
import logging
import sys

import meanvarkit

logger = logging.getLogger("normalize")


def build_parser():
    # Rutas de los archivos CSV de entrada
    parser = argparse.ArgumentParser(description="Normalizar matrices de caracteristicas")
    parser.add_argument("--reference", default="ref_features.csv")
    parser.add_argument("--holdout", default="eval_features.csv")
    return parser


def normalize_features(reference_rows, holdout_rows):
    normalizer = meanvarkit.MeanVarNormalizer()
    # Solo los datos de referencia definen media y varianza, nunca los de evaluacion
    normalizer.learn(reference_rows)
    normalized_ref = normalizer.apply(reference_rows)
    # La misma transformacion se aplica a los datos de evaluacion
    normalized_eval = normalizer.apply(holdout_rows)
    return normalized_ref, normalized_eval


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        reference_rows = meanvarkit.read_table(args.reference)
        holdout_rows = meanvarkit.read_table(args.holdout)
    except FileNotFoundError as exc:
        logger.error("Archivo no encontrado: %s", exc)
        return 1
    normalized_ref, normalized_eval = normalize_features(reference_rows, holdout_rows)
    logger.info("Filas de referencia: %d, de evaluacion: %d", len(normalized_ref), len(normalized_eval))
    return 0


if __name__ == "__main__":
    sys.exit(main())
````

</details>

<details><summary>Script modificado completo</summary>

````python
import argparse
import logging
import sys

from sklearn import preprocessing

logger = logging.getLogger("normalize")


def build_parser():
    # Rutas de los archivos CSV de entrada
    parser = argparse.ArgumentParser(description="Normalizar matrices de caracteristicas")
    parser.add_argument("--reference", default="ref_features.csv")
    parser.add_argument("--holdout", default="eval_features.csv")
    return parser


def normalize_features(reference_rows, holdout_rows):
    normalizer = preprocessing.Normalizer()
    # Solo los datos de referencia definen media y varianza, nunca los de evaluacion
    normalizer.learn(reference_rows)
    normalized_ref = normalizer.apply(reference_rows)
    # La misma transformacion se aplica a los datos de evaluacion
    normalized_eval = normalizer.apply(holdout_rows)
    return normalized_ref, normalized_eval


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        reference_rows = pd.read_csv(args.reference)
        holdout_rows = pd.read_csv(args.holdout)
    except FileNotFoundError as exc:
        logger.error("Archivo no encontrado: %s", exc)
        return 1
    normalized_ref, normalized_eval = normalize_features(reference_rows, holdout_rows)
    logger.info("Filas de referencia: %d, de evaluacion: %d", len(normalized_ref), len(normalized_eval))
    return 0


if __name__ == "__main__":
    sys.exit(main())
````

</details>

### Preservación de la tarea

¿El script modificado hace la misma tarea que el original? `si` / `parcial` / `no`.

- preserva: no
- que_se_perdio: se normaliza cada fila a norma L2 y no por media y varianza

### Faithfulness (heurística)

Identificadores de la API nueva en las líneas cambiadas. La heurística dice que un identificador es fiel si aparece en los chunks que recibió el LLM. Marcá los números en los que **la heurística se equivoca** (p. ej. aparece en el contexto pero no como esa API, o no aparece y sí está).

| # | Identificador | Línea | Tipo | ¿En el contexto? (heurística) | Dónde aparece |
|---|---|---|---|---|---|
| 1 | `sklearn` | 5 | module | sí | `This class is hence suitable for use in the early steps of a 'sklearn.pipeline.Pipeline':` |
| 2 | `preprocessing` | 5 | import | sí | `X_normalized = preprocessing.normalize(X, norm='l2')` |
| 3 | `Normalizer` | 19 | attr | sí | `The 'preprocessing' module further provides a utility class 'Normalizer' that implements the same operation using the 'Transformer' API (eve` |

- errores_heuristica: ninguno

## R16

- **Tarea del script:** Validar los argumentos de una función que repite un texto N veces con un separador, convirtiendo tipos cuando es posible y mostrando el error si la validación falla.
- **Import reemplazado:** `argcheckone`

### Cambios

````diff
--- original
+++ modificado
@@ -3,10 +3,15 @@
 import sys
 
-import argcheckone
+import typer
 
 logger = logging.getLogger("validacion")
 
 
-@argcheckone.enforce
+def name_callback(value: str):
+    if not isinstance(value, str):
+        raise typer.BadParameter("El valor debe ser una cadena")
+    return value
+
+@typer.Option(callback=name_callback)
 def repetir(texto: str, veces: int, *, separador: bytes = b"") -> bytes:
     """Repite el texto la cantidad de veces indicada, unido por el separador."""
````

<details><summary>Script original completo</summary>

````python
import argparse
import logging
import sys

import argcheckone

logger = logging.getLogger("validacion")


@argcheckone.enforce
def repetir(texto: str, veces: int, *, separador: bytes = b"") -> bytes:
    """Repite el texto la cantidad de veces indicada, unido por el separador."""
    codificado = texto.encode()
    return separador.join(codificado for _ in range(veces))


def ejemplos() -> None:
    print(repetir("hola", 3))
    print(repetir("x", "4", separador=b" "))
    try:
        repetir("hola", "incorrecto")
    except argcheckone.CheckError as error:
        logger.warning("Argumento inválido: %s", error)
        print(error)


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Demostración de validación de argumentos")
    parser.add_argument("--quiet", action="store_true", help="Solo mostrar advertencias")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    logging.basicConfig(level=logging.INFO)
    if args.quiet:
        logger.setLevel(logging.WARNING)
    ejemplos()
    return 0


if __name__ == "__main__":
    sys.exit(main())
````

</details>

<details><summary>Script modificado completo</summary>

````python
import argparse
import logging
import sys

import typer

logger = logging.getLogger("validacion")


def name_callback(value: str):
    if not isinstance(value, str):
        raise typer.BadParameter("El valor debe ser una cadena")
    return value

@typer.Option(callback=name_callback)
def repetir(texto: str, veces: int, *, separador: bytes = b"") -> bytes:
    """Repite el texto la cantidad de veces indicada, unido por el separador."""
    codificado = texto.encode()
    return separador.join(codificado for _ in range(veces))


def ejemplos() -> None:
    print(repetir("hola", 3))
    print(repetir("x", "4", separador=b" "))
    try:
        repetir("hola", "incorrecto")
    except argcheckone.CheckError as error:
        logger.warning("Argumento inválido: %s", error)
        print(error)


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Demostración de validación de argumentos")
    parser.add_argument("--quiet", action="store_true", help="Solo mostrar advertencias")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    logging.basicConfig(level=logging.INFO)
    if args.quiet:
        logger.setLevel(logging.WARNING)
    ejemplos()
    return 0


if __name__ == "__main__":
    sys.exit(main())
````

</details>

### Preservación de la tarea

¿El script modificado hace la misma tarea que el original? `si` / `parcial` / `no`.

- preserva: no
- que_se_perdio: se pierde la validacion y conversion de tipos, ademas de que se usa como decorador algo que no lo es

### Faithfulness (heurística)

Identificadores de la API nueva en las líneas cambiadas. La heurística dice que un identificador es fiel si aparece en los chunks que recibió el LLM. Marcá los números en los que **la heurística se equivoca** (p. ej. aparece en el contexto pero no como esa API, o no aparece y sí está).

| # | Identificador | Línea | Tipo | ¿En el contexto? (heurística) | Dónde aparece |
|---|---|---|---|---|---|
| 1 | `typer` | 5 | module | sí | `import typer` |
| 2 | `callback` | 15 | kwarg | sí | `def main(name: Annotated[str ¦ None, typer.Option(callback=name_callback)] = None):` |
| 3 | `Option` | 15 | attr | sí | `def main(name: Annotated[str ¦ None, typer.Option(callback=name_callback)] = None):` |
| 4 | `BadParameter` | 12 | attr | sí | `raise typer.BadParameter("Only Camila is allowed")` |

- errores_heuristica: ninguno

## R17

- **Tarea del script:** endpoint que sugiere los N nombres de ciudad más parecidos a una búsqueda mal escrita
- **Import reemplazado:** `fuzzywuzzy`

### Cambios

````diff
--- original
+++ modificado
@@ -4,5 +4,5 @@
 
 from flask import Flask, jsonify, request
-from fuzzywuzzy import process
+from rapidfuzz.process import extractOne
 
 app = Flask(__name__)
@@ -21,5 +21,5 @@
 def suggest(query, limit=MAX_SUGGESTIONS):
     """Return the closest city names to a (possibly misspelled) query."""
-    matches = process.extract(query, load_cities(), limit=limit)
+    matches = extractOne(query, load_cities(), limit=limit)
     return [{"city": name, "score": score} for name, score in matches if score >= MIN_SCORE]
````

<details><summary>Script original completo</summary>

````python
import json
import os
from functools import lru_cache

from flask import Flask, jsonify, request
from fuzzywuzzy import process

app = Flask(__name__)

CITIES_FILE = os.environ.get("CITIES_FILE", "data/cities.json")
MAX_SUGGESTIONS = 5
MIN_SCORE = 60


@lru_cache(maxsize=1)
def load_cities():
    with open(CITIES_FILE, encoding="utf-8") as fh:
        return [c["name"] for c in json.load(fh)]


def suggest(query, limit=MAX_SUGGESTIONS):
    """Return the closest city names to a (possibly misspelled) query."""
    matches = process.extract(query, load_cities(), limit=limit)
    return [{"city": name, "score": score} for name, score in matches if score >= MIN_SCORE]


@app.get("/api/cities/suggest")
def suggest_endpoint():
    query = request.args.get("q", "").strip()
    if len(query) < 2:
        return jsonify({"error": "query must have at least 2 characters"}), 400
    limit = min(int(request.args.get("limit", MAX_SUGGESTIONS)), 20)
    return jsonify({"query": query, "suggestions": suggest(query, limit)})


@app.get("/healthz")
def health():
    return {"status": "ok", "cities": len(load_cities())}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
````

</details>

<details><summary>Script modificado completo</summary>

````python
import json
import os
from functools import lru_cache

from flask import Flask, jsonify, request
from rapidfuzz.process import extractOne

app = Flask(__name__)

CITIES_FILE = os.environ.get("CITIES_FILE", "data/cities.json")
MAX_SUGGESTIONS = 5
MIN_SCORE = 60


@lru_cache(maxsize=1)
def load_cities():
    with open(CITIES_FILE, encoding="utf-8") as fh:
        return [c["name"] for c in json.load(fh)]


def suggest(query, limit=MAX_SUGGESTIONS):
    """Return the closest city names to a (possibly misspelled) query."""
    matches = extractOne(query, load_cities(), limit=limit)
    return [{"city": name, "score": score} for name, score in matches if score >= MIN_SCORE]


@app.get("/api/cities/suggest")
def suggest_endpoint():
    query = request.args.get("q", "").strip()
    if len(query) < 2:
        return jsonify({"error": "query must have at least 2 characters"}), 400
    limit = min(int(request.args.get("limit", MAX_SUGGESTIONS)), 20)
    return jsonify({"query": query, "suggestions": suggest(query, limit)})


@app.get("/healthz")
def health():
    return {"status": "ok", "cities": len(load_cities())}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
````

</details>

### Preservación de la tarea

¿El script modificado hace la misma tarea que el original? `si` / `parcial` / `no`.

- preserva: no
- que_se_perdio: devuelve solo una ciudad en vez de las N mas parecidas

### Faithfulness (heurística)

Identificadores de la API nueva en las líneas cambiadas. La heurística dice que un identificador es fiel si aparece en los chunks que recibió el LLM. Marcá los números en los que **la heurística se equivoca** (p. ej. aparece en el contexto pero no como esa API, o no aparece y sí está).

| # | Identificador | Línea | Tipo | ¿En el contexto? (heurística) | Dónde aparece |
|---|---|---|---|---|---|
| 1 | `rapidfuzz.process` | 6 | module | sí | `from rapidfuzz.process import extractOne` |
| 2 | `extractOne` | 6 | import | sí | `from rapidfuzz.process import extractOne` |
| 3 | `limit` | 23 | kwarg | no | — |

- errores_heuristica: ninguno

## R18

- **Tarea del script:** Dibujar un gráfico de barras con ruido aleatorio sobre una imagen de muestra colocada en la figura, con grilla y color fijo.
- **Import reemplazado:** `plotnimbus`

### Cambios

````diff
--- original
+++ modificado
@@ -3,5 +3,6 @@
 import sys
 
-from plotnimbus import Figure, load_sample_image
+import plotly.graph_objects as go
+from PIL import Image
 
 logger = logging.getLogger(__name__)
@@ -17,10 +18,21 @@
 
 def draw(xs, ys, image):
-    fig = Figure()
+    fig = go.Figure()
     ax = fig.add_axes()
     ax.bar(xs, ys, color="#6bbc6b")
     ax.grid(True)
     # Superponer la imagen de muestra en la esquina inferior izquierda
-    fig.place_image(image, 25, 25, zorder=3, alpha=0.7)
+    fig.add_layout_image(
+        dict(
+            source=image,
+            xref="x",
+            yref="y",
+            x=25,
+            y=25,
+            sizex=0.2,
+            sizey=0.2,
+            opacity=0.7,
+            layer="above")
+    )
     return fig
````

<details><summary>Script original completo</summary>

````python
import logging
import random
import sys

from plotnimbus import Figure, load_sample_image

logger = logging.getLogger(__name__)
SAMPLE_NAME = "logo2.png"


def noisy_bars(count, seed):
    rng = random.Random(seed)
    xs = list(range(count))
    ys = [x + rng.gauss(0.0, 1.0) for x in xs]
    return xs, ys


def draw(xs, ys, image):
    fig = Figure()
    ax = fig.add_axes()
    ax.bar(xs, ys, color="#6bbc6b")
    ax.grid(True)
    # Superponer la imagen de muestra en la esquina inferior izquierda
    fig.place_image(image, 25, 25, zorder=3, alpha=0.7)
    return fig


def main():
    logging.basicConfig(level=logging.INFO)
    with load_sample_image(SAMPLE_NAME) as handle:
        image = handle.read_pixels()
    xs, ys = noisy_bars(30, seed=19680801)
    fig = draw(xs, ys, image)
    logger.info("Figura construida con %d barras", len(xs))
    fig.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
````

</details>

<details><summary>Script modificado completo</summary>

````python
import logging
import random
import sys

import plotly.graph_objects as go
from PIL import Image

logger = logging.getLogger(__name__)
SAMPLE_NAME = "logo2.png"


def noisy_bars(count, seed):
    rng = random.Random(seed)
    xs = list(range(count))
    ys = [x + rng.gauss(0.0, 1.0) for x in xs]
    return xs, ys


def draw(xs, ys, image):
    fig = go.Figure()
    ax = fig.add_axes()
    ax.bar(xs, ys, color="#6bbc6b")
    ax.grid(True)
    # Superponer la imagen de muestra en la esquina inferior izquierda
    fig.add_layout_image(
        dict(
            source=image,
            xref="x",
            yref="y",
            x=25,
            y=25,
            sizex=0.2,
            sizey=0.2,
            opacity=0.7,
            layer="above")
    )
    return fig


def main():
    logging.basicConfig(level=logging.INFO)
    with load_sample_image(SAMPLE_NAME) as handle:
        image = handle.read_pixels()
    xs, ys = noisy_bars(30, seed=19680801)
    fig = draw(xs, ys, image)
    logger.info("Figura construida con %d barras", len(xs))
    fig.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
````

</details>

### Preservación de la tarea

¿El script modificado hace la misma tarea que el original? `si` / `parcial` / `no`.

- preserva: no
- que_se_perdio: metodos incorrectos para los Figure de plotly

### Faithfulness (heurística)

Identificadores de la API nueva en las líneas cambiadas. La heurística dice que un identificador es fiel si aparece en los chunks que recibió el LLM. Marcá los números en los que **la heurística se equivoca** (p. ej. aparece en el contexto pero no como esa API, o no aparece y sí está).

| # | Identificador | Línea | Tipo | ¿En el contexto? (heurística) | Dónde aparece |
|---|---|---|---|---|---|
| 1 | `plotly.graph_objects` | 5 | module | sí | `import plotly.graph_objects as go` |
| 2 | `PIL` | 6 | module | sí | `A background image can be added to the layout of a figure with 'fig.add_layout_image' or by setting the 'images' parameter of 'go.Layout'. T` |
| 3 | `Image` | 6 | import | sí | `#### Add a Background Image` |
| 4 | `Figure` | 20 | attr | sí | `- 'matplotlib.figure.Figure.add_axes'` |
| 5 | `add_layout_image` | 25 | attr | sí | `A background image can be added to the layout of a figure with 'fig.add_layout_image' or by setting the 'images' parameter of 'go.Layout'. T` |

- errores_heuristica: ninguno

## R19

- **Tarea del script:** Generar variantes de una imagen aplicando distintas correcciones gamma a una copia del original, sin modificar el archivo fuente.
- **Import reemplazado:** `gammaspin`

### Cambios

````diff
--- original
+++ modificado
@@ -4,5 +4,5 @@
 from pathlib import Path
 
-import gammaspin
+from wand.image import Image
 
 logger = logging.getLogger("variantes_gamma")
@@ -13,10 +13,10 @@
     """Aplica cada valor gamma a una copia de la imagen original."""
     output_dir.mkdir(parents=True, exist_ok=True)
-    with gammaspin.Picture(filename=str(source)) as original:
+    with Image(filename=str(source)) as original:
         for gamma in gammas:
-            with gammaspin.Picture(original) as copy:
-                copy.apply_gamma(gamma)
+            with original.clone() as copy:
+                copy.gamma(gamma)
                 target = output_dir / f"gamma_{gamma}.png"
-                copy.save_to(target)
+                copy.save(filename=str(target))
                 logger.info("Variante con gamma %.2f guardada en %s", gamma, target)
````

<details><summary>Script original completo</summary>

````python
import argparse
import logging
import sys
from pathlib import Path

import gammaspin

logger = logging.getLogger("variantes_gamma")
GAMMA_VALUES = [0.8, 0.9, 1.33, 1.66]


def generate_variants(source: Path, output_dir: Path, gammas=GAMMA_VALUES) -> None:
    """Aplica cada valor gamma a una copia de la imagen original."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with gammaspin.Picture(filename=str(source)) as original:
        for gamma in gammas:
            with gammaspin.Picture(original) as copy:
                copy.apply_gamma(gamma)
                target = output_dir / f"gamma_{gamma}.png"
                copy.save_to(target)
                logger.info("Variante con gamma %.2f guardada en %s", gamma, target)


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Variantes gamma de una imagen")
    parser.add_argument("source", type=Path, help="Imagen de entrada")
    parser.add_argument("--out", type=Path, default=Path("variantes"))
    return parser.parse_args(argv)


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO)
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if not args.source.exists():
        logger.error("No existe la imagen %s", args.source)
        return 1
    generate_variants(args.source, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
````

</details>

<details><summary>Script modificado completo</summary>

````python
import argparse
import logging
import sys
from pathlib import Path

from wand.image import Image

logger = logging.getLogger("variantes_gamma")
GAMMA_VALUES = [0.8, 0.9, 1.33, 1.66]


def generate_variants(source: Path, output_dir: Path, gammas=GAMMA_VALUES) -> None:
    """Aplica cada valor gamma a una copia de la imagen original."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with Image(filename=str(source)) as original:
        for gamma in gammas:
            with original.clone() as copy:
                copy.gamma(gamma)
                target = output_dir / f"gamma_{gamma}.png"
                copy.save(filename=str(target))
                logger.info("Variante con gamma %.2f guardada en %s", gamma, target)


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Variantes gamma de una imagen")
    parser.add_argument("source", type=Path, help="Imagen de entrada")
    parser.add_argument("--out", type=Path, default=Path("variantes"))
    return parser.parse_args(argv)


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO)
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if not args.source.exists():
        logger.error("No existe la imagen %s", args.source)
        return 1
    generate_variants(args.source, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
````

</details>

### Preservación de la tarea

¿El script modificado hace la misma tarea que el original? `si` / `parcial` / `no`.

- preserva: si
- que_se_perdio:

### Faithfulness (heurística)

Identificadores de la API nueva en las líneas cambiadas. La heurística dice que un identificador es fiel si aparece en los chunks que recibió el LLM. Marcá los números en los que **la heurística se equivoca** (p. ej. aparece en el contexto pero no como esa API, o no aparece y sí está).

| # | Identificador | Línea | Tipo | ¿En el contexto? (heurística) | Dónde aparece |
|---|---|---|---|---|---|
| 1 | `wand.image` | 6 | module | sí | `The 'wand.image.Image.level()' method can also adjust 'gamma' value.` |
| 2 | `Image` | 6 | import | sí | `The 'wand.image.Image.level()' method can also adjust 'gamma' value.` |
| 3 | `filename` | 15 | kwarg | sí | `with Image(filename='enhancement.jpg') as img_src:` |
| 4 | `clone` | 17 | attr | sí | `If you have an image already and have to copy it for safe manipulation, use 'wand.image.Image.clone()' method:` |
| 5 | `gamma` | 18 | attr | sí | `Gamma correction allows you to adjust the luminance of an image. Resulting pixels are defined as 'pixel^(1/gamma)'. The value of 'gamma' is ` |
| 6 | `save` | 20 | attr | sí | `img.save(filename='enhancement-level.jpg')` |

- errores_heuristica: ninguno

## R20

- **Tarea del script:** Definir un paso de ciclo de vida para un proceso trabajador que se inicializa con sus argumentos, se activa al arrancar, se desactiva al apagarse y registra su terminación.
- **Import reemplazado:** `plugsteps`

### Cambios

````diff
--- original
+++ modificado
@@ -3,11 +3,11 @@
 import sys
 
-import plugsteps
+from celery import bootsteps
 
 logger = logging.getLogger("paso_trabajador")
 
 
-class ExampleWorkerStep(plugsteps.LifecycleStep):
-    requires = {"workers.components:Pool"}
+class ExampleWorkerStep(bootsteps.StartStopStep):
+    requires = {'celery.worker.components:Pool'}
 
     def __init__(self, worker, **kwargs):
````

<details><summary>Script original completo</summary>

````python
import json
import logging
import sys

import plugsteps

logger = logging.getLogger("paso_trabajador")


class ExampleWorkerStep(plugsteps.LifecycleStep):
    requires = {"workers.components:Pool"}

    def __init__(self, worker, **kwargs):
        print("Se llama cuando se construye el controlador del trabajador")
        print("Argumentos del controlador: " + json.dumps(kwargs, default=str))

    def create(self, worker):
        # este método puede delegar las acciones
        # en otro objeto que implemente start y stop.
        return self

    def start(self, worker):
        print("Se llama cuando el trabajador arranca.")

    def stop(self, worker):
        print("Se llama cuando el trabajador se apaga.")

    def terminate(self, worker):
        print("Se llama cuando el trabajador termina de forma forzada.")


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    plugsteps.register(ExampleWorkerStep)
    logger.info("Paso de trabajador registrado")
    return 0


if __name__ == "__main__":
    sys.exit(main())
````

</details>

<details><summary>Script modificado completo</summary>

````python
import json
import logging
import sys

from celery import bootsteps

logger = logging.getLogger("paso_trabajador")


class ExampleWorkerStep(bootsteps.StartStopStep):
    requires = {'celery.worker.components:Pool'}

    def __init__(self, worker, **kwargs):
        print("Se llama cuando se construye el controlador del trabajador")
        print("Argumentos del controlador: " + json.dumps(kwargs, default=str))

    def create(self, worker):
        # este método puede delegar las acciones
        # en otro objeto que implemente start y stop.
        return self

    def start(self, worker):
        print("Se llama cuando el trabajador arranca.")

    def stop(self, worker):
        print("Se llama cuando el trabajador se apaga.")

    def terminate(self, worker):
        print("Se llama cuando el trabajador termina de forma forzada.")


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    plugsteps.register(ExampleWorkerStep)
    logger.info("Paso de trabajador registrado")
    return 0


if __name__ == "__main__":
    sys.exit(main())
````

</details>

### Preservación de la tarea

¿El script modificado hace la misma tarea que el original? `si` / `parcial` / `no`.

- preserva: no
- que_se_perdio: falla por llamado a un metodo que ya no esta importado

### Faithfulness (heurística)

Identificadores de la API nueva en las líneas cambiadas. La heurística dice que un identificador es fiel si aparece en los chunks que recibió el LLM. Marcá los números en los que **la heurística se equivoca** (p. ej. aparece en el contexto pero no como esa API, o no aparece y sí está).

| # | Identificador | Línea | Tipo | ¿En el contexto? (heurística) | Dónde aparece |
|---|---|---|---|---|---|
| 1 | `celery` | 5 | module | sí | `from celery import bootsteps` |
| 2 | `bootsteps` | 5 | import | sí | `from celery import bootsteps` |
| 3 | `StartStopStep` | 10 | attr | sí | `class ExampleWorkerStep(bootsteps.StartStopStep):` |

- errores_heuristica: ninguno
