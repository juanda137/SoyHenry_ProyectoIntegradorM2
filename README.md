# Bot de FAQs para HR con RAG

Chatbot de soporte para una plataforma de HR SaaS ("Bot para HR"). El equipo
de RR. HH. recibe más de 200 preguntas repetitivas por día —cómo pedir
vacaciones, cómo configurar el SSO, cuánto se conservan los recibos de
nómina— que ya están documentadas en una FAQ de texto plano. Este proyecto
responde esas preguntas automáticamente: recupera los fragmentos relevantes
del documento y genera una respuesta fundamentada en ellos. Devuelve un JSON
con exactamente tres claves (`user_question`, `system_answer`,
`chunks_related`) para que otro sistema lo consuma sin transformaciones, y
tiene un agente evaluador opcional que puntúa cada respuesta de 0 a 10.

Es un repositorio git propio, entrega del **Módulo 2** de la carrera de AI
Engineering. Autocontenido: no importa código de `primeros_pasos_ai_engineer`
ni de `ai-engineer-lab`, aunque reutiliza sus ideas.

---

## Por qué RAG

**Esto usa RAG (Retrieval-Augmented Generation) porque primero se recuperan
los chunks relevantes del documento y después se genera la respuesta usando
sólo esos chunks**, en vez de confiarle todo a lo que el modelo "recuerda".

```
Indexación (una sola vez, o cada vez que cambia el documento)
  data/faq_document.txt  →  chunking por P/R  →  embeddings  →  index/index.json

Consulta (una por pregunta)
  pregunta  →  embeber  →  buscar (híbrida: coseno + BM25 + RRF)
            →  2-5 chunks como contexto  →  LLM genera con ESE contexto  →  JSON
```

Beneficios de separar recuperar de generar:

- **Conocimiento actualizable sin reentrenar**: si cambia una política de
  RR. HH., se edita `data/faq_document.txt` y se vuelve a correr
  `python -m src.build_index`. El modelo no cambia.
- **Atribución de fuentes**: cada afirmación cita `[chunk N]` en el texto, y
  `chunks_related` trae esos mismos chunks completos, con su similitud.
- **Transparencia y auditabilidad**: se puede revisar exactamente qué texto
  vio el modelo antes de responder, en vez de confiar a ciegas.
- **Menos tokens por pedido**: el prompt lleva 2 a 5 fragmentos (unos pocos
  cientos de tokens), no el documento entero. Con un documento más grande
  esto deja de ser opcional: hay un límite de tokens por pedido.

---

## Requisitos

- **Python 3.14** (versión con la que se probó este proyecto).
- Una clave de API: **Gemini** (tiene capa gratuita, recomendada) u
  **OpenAI**.

---

## Instalación

1. Crear el entorno virtual:

   ```bash
   python -m venv .venv
   ```

2. Activarlo:

   ```bash
   .venv\Scripts\activate
   ```

   En Linux/macOS: `source .venv/bin/activate`

3. Instalar las dependencias:

   ```bash
   pip install -r requirements.txt
   ```

4. Configurar la clave de API. Hay dos formas:

   **a) Con un archivo `.env`** (recomendado para desarrollo):

   ```bash
   copy .env.example .env
   ```

   En Linux/macOS: `cp .env.example .env`

   Y completar `GEMINI_API_KEY` (o `OPENAI_API_KEY`) dentro de `.env`.

   **b) Con variables de entorno**, sin archivo:

   ```bash
   # Linux / macOS
   export LLM_PROVIDER=openai   # o gemini; elige el proveedor
   export GEMINI_API_KEY=tu-clave
   export OPENAI_API_KEY=tu-clave
   ```

   ```cmd
   :: Windows (cmd)
   set LLM_PROVIDER=openai
   set GEMINI_API_KEY=tu-clave
   ```

   ```powershell
   # Windows (PowerShell)
   $env:LLM_PROVIDER="openai"
   $env:GEMINI_API_KEY="tu-clave"
   ```

   `LLM_PROVIDER` elige el proveedor (`gemini` por defecto, u `openai`).

Las claves nunca se leen del código: en todo el proyecto se resuelven con
`os.getenv()`, con esta prioridad: **flag de CLI → variable de entorno →
valor por defecto**.

**Por qué cambiar de proveedor es tan simple**: el SDK `openai` es un
cliente del protocolo `/chat/completions` y `/embeddings`, no de la empresa.
Gemini expone ese mismo protocolo en una URL de compatibilidad. Cambiar de
proveedor es sólo cambiar `LLM_PROVIDER` + la clave correspondiente; el
código no cambia una línea.

- Clave de Gemini (capa gratuita): https://aistudio.google.com/apikey
- Clave de OpenAI: https://platform.openai.com/api-keys

---

## Uso

Construir el índice (lee `data/faq_document.txt`, genera los embeddings y
guarda `index/index.json`):

```bash
python -m src.build_index
```

Salida real de esa corrida (Gemini, `gemini-embedding-001`):

```
Índice guardado en .../proyecto_2/index/index.json
  chunks: 30
  tokens por chunk: mínimo 122, máximo 176, promedio 143
  embeddings: 30 vectores de 3072 dimensiones (gemini-embedding-001)
  cobertura del documento: 100% (verificada antes de guardar)
```

Consultar (`main.py` construye el índice solo si todavía no existe):

```bash
python main.py "¿Puedo cancelar una solicitud de vacaciones que ya fue aprobada?"
```

Salida real (recortada) para esa pregunta, tomada de
`outputs/sample_queries.json`:

```json
{
  "user_question": "¿Puedo cancelar una solicitud de vacaciones que ya fue aprobada?",
  "system_answer": "Sí. Podés cancelarla hasta 48 horas antes del inicio, desde Ausencias > Mis solicitudes > Cancelar [chunk 9]. Fuera de ese plazo ya no podés hacerlo por tu cuenta, sino que debés pedírselo al administrador de RR. HH. para que la cancele manualmente [chunk 9].",
  "chunks_related": [
    {"id": 9, "seccion": "Vacaciones y ausencias", "similitud": 0.8905, "texto": "..."},
    {"id": 7, "seccion": "Vacaciones y ausencias", "similitud": 0.7336, "texto": "..."},
    {"id": 11, "seccion": "Vacaciones y ausencias", "similitud": 0.7019, "texto": "..."}
  ]
}
```

Con el SDK de OpenAI en vez de Gemini, sólo cambia la clave y el proveedor:

```bash
export LLM_PROVIDER=openai
export OPENAI_API_KEY=your-key
python main.py "¿Cómo solicito vacaciones?"
```

Al cambiar de proveedor cambia también el modelo de embeddings, así que
`main.py` reindexa solo la primera vez que lo notás: si ya existe
`index/index.json` pero quedó hecho con el modelo del proveedor anterior,
lo reconstruye automáticamente (y avisa por qué, por stderr) antes de
responder.

Otras opciones de `main.py` / `python -m src.query`:

```bash
# Sólo búsqueda semántica (sin BM25 ni RRF), para comparar contra la híbrida
python main.py "¿Cómo solicito vacaciones?" --metodo knn

# Suma el agente evaluador; el puntaje y la razón salen por stderr
python main.py "¿Cómo solicito vacaciones?" --evaluar
```

Generar los tres archivos de `outputs/` (cinco preguntas fijas, corre todo
el pipeline de punta a punta):

```bash
python -m src.generar_ejemplos
```

Correr los tests (offline, no llaman a ninguna API):

```bash
python -m pytest tests/ -v
```

La primera corrida descarga el tokenizador de `tiktoken` (necesita conexión
a internet una sola vez); después queda en caché local y las corridas
siguientes son completamente offline.

---

## Estructura del proyecto

```
proyecto_2/
├── data/
│   └── faq_document.txt        FAQ de "Bot para HR", ~2865 palabras
├── src/
│   ├── config.py                proveedores, modelos y parámetros
│   ├── llm_client.py            embeddings + chat, con reintentos y timeout
│   ├── chunking.py               cargar y partir el documento en chunks
│   ├── build_index.py           pipeline de indexación (main)
│   ├── busqueda.py               coseno, BM25, RRF, filtro por rango, top-k
│   ├── query.py                  pipeline de consulta (main)
│   ├── evaluador.py              agente evaluador (LLM como juez)
│   └── generar_ejemplos.py      genera los tres archivos de outputs/
├── prompts/
│   ├── respuesta.txt             prompt del modelo que responde
│   └── evaluador.txt             prompt del agente evaluador
├── index/                        índice generado, fuera de git (.gitignore)
├── outputs/
│   ├── sample_queries.json       5 preguntas de ejemplo con su respuesta real
│   ├── sample_evaluations.json   evaluación real de cada respuesta
│   └── metricas_recuperacion.json  knn vs híbrida, medido
├── tests/                        pytest, 73 tests, todos offline
├── main.py                       atajo: indexa si falta y consulta
├── README.md
├── .env.example
├── .gitignore
└── requirements.txt
```

---

## Decisiones técnicas

### Chunking por estructura

El documento es una FAQ: cada par pregunta-respuesta ya es una unidad de
significado completa, así que el corte natural es ese par, no un tamaño fijo
que podría separar la pregunta de su propia respuesta.

Parámetros: `CHUNK_MIN_TOKENS=50` (un bloque más chico se une al vecino de
su misma sección), `CHUNK_MAX_TOKENS=500` (uno más grande se parte por
oraciones), `SOLAPAMIENTO_ORACIONES=1` (una oración repetida entre pedazos,
para no perder el hilo).

Resultado medido sobre el documento actual (`data/faq_document.txt`, con
`chunking.partir_en_chunks`, sin llamar a ninguna API):

| Métrica | Valor |
|---|---|
| Chunks | 30 |
| Tokens por chunk | mínimo 122, máximo 176, promedio 143 |
| Cobertura del documento | 100% (ninguna oración quedó fuera de todos los chunks) |

En el lab del Módulo 2 (`ai-engineer-lab`), sobre un manual chico, el
chunking por **tamaño fijo empató** con las estrategias avanzadas: con pocos
fragmentos no hay suficiente ruido para que la diferencia se note. Acá,
aunque el documento también es chico, chunking por estructura sigue
conviniendo por una razón distinta a la del volumen: una FAQ tiene una unidad
de significado obvia (el par P/R), y cortarla por tamaño fijo arriesgaría
partir una pregunta de su respuesta a la mitad, lo cual rompería el sentido
del fragmento sin importar cuántos fragmentos haya.

### Búsqueda híbrida

La búsqueda semántica (embeddings + coseno explícito, `A·B / (|A|·|B|)` con
NumPy) entiende sinónimos y paráfrasis, pero puede traer chunks "del mismo
tema" sin la palabra exacta que se preguntó. La léxica (BM25) premia
coincidencias exactas como "SSO", "IBAN" o "Slack", pero no entiende
sinónimos. Se combinan con RRF (Reciprocal Rank Fusion, `RRF_K=60`): como el
coseno y BM25 usan escalas distintas, RRF ignora los puntajes y fusiona por
la *posición* de cada chunk en cada ranking.

Después de fusionar, un filtro por rango descarta lo que no llega a
`SIMILITUD_MINIMA = 0.68` (calibrado en la Tarea 8: los chunks que
efectivamente responden las preguntas de ejemplo tenían coseno ≥ 0.836, y la
pregunta fuera de alcance no superaba 0.5211; el umbral quedó en el punto
medio, redondeado). Se devuelven entre `TOP_K_MIN=2` y `TOP_K_MAX=5` chunks.

**Este umbral está calibrado específicamente para `gemini-embedding-001`.**
Con otro modelo de embeddings (por ejemplo `text-embedding-3-small` de
OpenAI) la escala del coseno es distinta, así que `0.68` no es portable sin
más: hay que recalibrarlo corriendo `python -m src.generar_ejemplos` con ese
modelo y aplicando el mismo método A/B (comparar el coseno mínimo de los
chunks que sí responden contra el máximo de la pregunta fuera de alcance).

**El filtro por rango se combina con un mínimo de `TOP_K_MIN=2` chunks.**
Cuando nada pasa el umbral —el caso real es la pregunta sobre la receta de
la paella valenciana, con similitud máxima 0.52, bien por debajo de 0.68— el
sistema igual devuelve los 2 mejores chunks encontrados (aunque no sirvan
para responder). No es el filtro el que produce el "no encontré esa
información": es el prompt (`prompts/respuesta.txt`), que instruye a
responder sólo con el contexto dado y usar una frase fija si ese contexto no
alcanza. El filtro decide *cuántos* chunks van al prompt; la honestidad de
la respuesta es responsabilidad del prompt.

Resultado medido (`outputs/metricas_recuperacion.json`, % de chunks
devueltos que contienen al menos una palabra clave de la pregunta, promedio
sobre las 4 preguntas dentro de alcance):

| Método | % con palabras clave |
|---|---|
| knn (sólo semántica) | 100.0% |
| híbrida (coseno + BM25 + RRF) | 100.0% |

**Los dos métodos empataron en 100%, y hay que decirlo tal cual: la híbrida
no ganó en esta métrica.** La razón es que la métrica es deliberadamente
laxa: alcanza con que **una sola** palabra clave de la pregunta aparezca en
el chunk para que cuente como acierto, y con un documento de 30 chunks bien
escritos la búsqueda semántica sola ya encuentra ese mínimo casi siempre. Aun
así se mantiene la híbrida como método por defecto, porque su valor no se ve
en este promedio: BM25 rescata coincidencias **exactas** de términos como
"SSO", "IBAN" o "Slack" que un embedding podría diluir entre chunks
temáticamente parecidos, y ese caso no está representado entre las 5
preguntas de ejemplo actuales. Es una apuesta a la robustez fuera de esta
muestra, no un número que la muestra actual pueda demostrar.

### Sección como contexto del embedding

Lo que se embebe no es sólo el texto del chunk, sino
`"Sección: <sección>. <texto>"` (técnica de L1-04 del lab): la metadata
como contexto ayuda a que el vector capture también de qué área del FAQ
viene el fragmento. El texto guardado y mostrado al usuario queda intacto,
sin ese prefijo.

### Tokens con tiktoken

Los tokens se cuentan con `tiktoken` (`cl100k_base`), el tokenizador de los
modelos de OpenAI. Es una medida local, gratuita y estándar para verificar
el rango de 50 a 500 tokens; con Gemini el conteo exacto varía un poco
respecto al tokenizador real del modelo, pero sirve igual como vara de
medir consistente. La primera vez que se usa, `tiktoken` descarga el
tokenizador (necesita conexión a internet); después queda en caché local.

### Evaluador

El agente evaluador (`src/evaluador.py`) es un LLM como juez, con
`temperature=0` para que sea lo más estable posible. Puntúa tres
dimensiones de 0 a 10 (`relevancia`, `fidelidad`, `completitud`) y da un
`reason`. El **puntaje final lo calcula Python**, no el modelo: promedio
redondeado de las tres dimensiones, con un techo — si `fidelidad < 5`, el
puntaje no puede superar la fidelidad, porque una respuesta que inventa no
se compensa con ser completa.

Puntajes reales (`outputs/sample_evaluations.json`, con
`SIMILITUD_MINIMA=0.68` ya calibrado):

| Pregunta | Score |
|---|---|
| si/no (cancelar vacaciones aprobadas) | 9 |
| cómo hacer (SSO con Entra ID) | 7 |
| dato exacto (días trasladables) | 7 |
| dos partes (formatos + retención de nómina) | 10 |
| fuera de alcance (paella valenciana) | 7 |

**El 7 de la pregunta fuera de alcance no es un defecto de la respuesta.**
Rechazar responder la receta de la paella es lo correcto ahí, y el prompt del
evaluador ya lo puntúa así: `fidelidad` y `completitud` dan 10 (la respuesta
no inventa nada y es tan completa y honesta como puede serlo un rechazo). Lo
que baja el promedio es `relevancia` (0 en esta pregunta), que mide qué tan
relacionados están los chunks recuperados con la pregunta — y ningún chunk
del FAQ de HR habla de paella, así que esa relevancia es legítimamente baja.
El score refleja eso, no un prompt más permisivo ni una respuesta peor.

Cuando el juez responde un JSON inválido o incompleto, se reintenta una vez
con otra temperatura (`TEMPERATURA_DE_REINTENTO=0.4`). Con `temperature=0`
el modelo es determinístico: si el corte viene de que el modelo se quedó sin
`MAX_TOKENS` (razonando antes de escribir el JSON), repetir la misma
pregunta a la misma temperatura reproduce el mismo corte exacto, así que el
reintento cambia la temperatura para tener una oportunidad distinta. **Es
una segunda oportunidad, no una garantía**: si el límite real es
`MAX_TOKENS=1200`, cambiar la temperatura no lo agranda, y el reintento
puede fallar por la misma razón. Si los dos intentos fallan, se informa "no
se pudo evaluar" — nunca se inventa un puntaje.

El README no puede dejar de mencionar el **sesgo de que un modelo se
evalúe a sí mismo**: por defecto el juez (`EVALUATOR_MODEL`) es el mismo
modelo que generó la respuesta, y un modelo tiende a ser indulgente con su
propio trabajo. Configurar `EVALUATOR_MODEL` a un modelo distinto mitiga
esto, a costa de otra llamada a otro proveedor o modelo.

### Sobre el modelo usado en los ejemplos

Los `outputs/*.json` de este repositorio se generaron con
**`gemini-3.5-flash-lite`**, no con `gemini-3.5-flash` (el modelo de chat
que documenta el spec original). Durante el desarrollo, la capa gratuita de
`gemini-3.5-flash` se agotó (errores `429 RateLimitError` repetidos, en
varias corridas espaciadas en minutos, mientras los embeddings seguían
funcionando sin problema — es decir, específicamente la cuota de chat). Al
cambiar a `gemini-3.5-flash-lite` las corridas terminaron limpias, sin
reintentos ni fallas. Por eso `src/config.py` tiene `gemini-3.5-flash-lite`
como modelo de chat por defecto para el proveedor `gemini`.

---

## Manejo de errores

- **Falta la clave de API**: mensaje claro al inicio (qué variable falta,
  cómo conseguirla), sin traceback. Ver `config.leer_clave`.
- **Falta el índice, o es de otro modelo de embeddings**: mensaje explicando
  qué correr (`python -m src.build_index`). Embeddings de modelos distintos
  no son comparables entre sí, así que esto se verifica antes de buscar.
  `main.py` construye el índice si todavía no existe, y lo reconstruye si
  quedó hecho con otro modelo de embeddings.
- **El proveedor falla** (errores 5xx, timeout, límite de uso): se
  reintenta hasta `REINTENTOS=3` veces, con espera que se duplica en cada
  intento (5s, 10s). Si se agotan los reintentos, el JSON de salida **igual
  respeta el contrato de 3 claves**: `system_answer` explica el problema y
  `chunks_related` trae lo que sí se recuperó; el proceso termina con código
  de salida distinto de 0.

---

## Limitaciones

Fuera de alcance a propósito (según el documento de diseño del proyecto):

- **Sin interfaz web ni chat interactivo**: es una CLI, pensada para
  integrarse a otro sistema vía el JSON de stdout.
- **Sin historial de conversación**: cada pregunta es independiente; no hay
  memoria entre llamadas.
- **Sin base vectorial externa** (ChromaDB, FAISS, etc.): con ~30 chunks,
  la búsqueda exacta con NumPy sobre toda la matriz tarda milisegundos. Un
  índice ANN (aproximado) resuelve un problema de escala que este documento
  no tiene; agregarlo acá sería complejidad sin beneficio medible.
- **Sin reranking con cross-encoder ni métricas de costo**: quedaron fuera
  del alcance de esta entrega.
