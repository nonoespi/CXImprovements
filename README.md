CX Improvements – Demo conversacional (Streamlit)

Aplicación Streamlit local que simula un chatbot conversacional para identificar una BU y/o un micromomento a partir de listas predefinidas y datos (Azure SQL o ficheros parquet). Una vez fijado el contexto, recopila el histórico de Improvements y arranca un segundo chatbot de análisis e inspiración (Azure OpenAI), con exportación a PDF del historial.

Funcionalidades clave

Selector de BU simulada (en la barra lateral) a partir de una lista fija.

Flujo 100% conversacional por botones (sin escribir) para:

Elegir solo BU,

Elegir solo micromomento (para todas las BUs compatibles),

Elegir micromomento en una BU concreta.
El flujo y el historial de conversación se renderizan en pantalla y se guardan para exportar.

Auto-selección de BU cuando un micromomento existe en una sola BU (evita preguntas redundantes).

Dos modos de datos:

ONLINE: lectura desde Azure SQL.

OFFLINE: lectura desde parquet (micromomentos_actuar.parquet, mejorasactuar.parquet).

Chat de análisis con Azure OpenAI para generar resúmenes e ideas inspiradas.

Descarga de PDF con toda la conversación (incluye ambos chatbots).

Arquitectura (alto nivel)

app.py: única app Streamlit con:

Carga de variables de entorno y detección de modo OFFLINE/ONLINE mediante SQL_DIALECT (offline → parquet; otro valor → SQL).

Funciones para cargar parquet/SQL, resolver equivalencias de micromomento ↔ micromomento_global, y montar los tres casos de filtrado (solo BU, solo micromomento, micromomento+BU).

UI conversacional por botones (bloques) y chat de análisis con Azure OpenAI.

Exportación a PDF del histórico.

Estructura de proyecto (sugerida)

.
├─ app.py
├─ requirements.txt
├─ .env # No se sube a Git
├─ data/
│ ├─ micromomentos_actuar.parquet
│ └─ mejorasactuar.parquet
└─ README.md

(La app también busca los parquet en la raíz si no existe ./data/)

Requisitos

Python 3.10–3.11 (recomendado).

Dependencias (vía requirements.txt): streamlit, pandas, sqlalchemy, pyodbc, python-dotenv, openai, reportlab, pyarrow, tiktoken, etc.
Nota: si vas a conectarte por TDS/TLS, considera python-tds y sqlalchemy-pytds además del driver ODBC cuando aplique.

Instalación
Opción A) Conda (recomendado para Windows)

conda deactivate
conda env remove --name cx_improvement_sep25 -y # opcional
conda create -n cx_improvement_sep25 python=3.11 streamlit pandas pyodbc sqlalchemy openai
conda activate cx_improvement_sep25
pip install --upgrade pip
pip install -r requirements.txt # para alinear con el repo

(Instala paquetes faltantes como reportlab, pyarrow, tiktoken, python-dotenv, sqlalchemy-pytds, python-tds, según tu modo ONLINE/OFFLINE.)

Opción B) Pip (entorno virtual)

python -m venv .venv

Windows: .venv\Scripts\activate
macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt

Configuración (.env)

Crea un fichero .env en la raíz (no lo subas a Git):

Azure OpenAI

AZURE_OPENAI_API_KEY=xxxxx
AZURE_OPENAI_ENDPOINT=https://xxxxx.openai.azure.com/

AZURE_OPENAI_DEPLOYMENT=gpt-4.1-nano
AZURE_OPENAI_API_VERSION=2024-05-01-preview

Azure SQL

SQL_DIALECT=pyodbc # 'offline' para parquet; 'pyodbc' o 'pytds' para SQL
SQL_DRIVER=ODBC Driver 17 for SQL Server
SQL_SERVER=stg-srv.database.windows.net
SQL_USERNAME=xxxx
SQL_PASSWORD=xxxx
SQL_DATABASE=xxxx

La app prioriza st.secrets si estás en Streamlit Cloud; para local usa .env. El modo OFFLINE se activa si SQL_DIALECT=offline.

Datos necesarios

Micromomentos: micromomentos_actuar.parquet con columnas BU, Micromomento, Micromomento_Global (la app puede normalizar a bu, micromomento, micromomento_global).

Improvements: mejorasactuar.parquet con (idealmente) FECHA, BU, MICROMOMENTO, MICROMOMENTO_GLOBAL, MEJORA, USUARIO.

Si vienen TITULO/DETALLE en lugar de MEJORA, la app construye MEJORA = TITULO + ': ' + DETALLE.

Coloca estos ficheros en ./data/ o en la raíz del proyecto.

Ejecución

streamlit run app.py

Barra lateral:

Selecciona la BU simulada y pulsa “Comenzar”.

Botón “Resetear demo” para limpiar estado y cachés.

Pantalla principal:

Mensaje de bienvenida y bloques de botones para seleccionar BU y/o micromomento.

Si eliges micromomento y solo pertenece a una BU, la app auto-confirma esa BU y continúa.

Tras confirmar el caso (solo BU / solo MM / MM+BU) el sistema muestra “Recopilando histórico…” y arranca el chat de análisis.

Descarga PDF de la conversación: botón/ícono de descarga en la cabecera.

Lógica de filtrado (resumen)

La app resuelve los tres casos con una única función de estado:

Micromomento para TODAS las BUs → filtra por MICROMOMENTO_GLOBAL.

Micromomento + BU concreta → filtra por BU y MICROMOMENTO_GLOBAL.

Solo BU → filtra por BU.

La equivalencia micromomento (por BU) → micromomento_global se resuelve con el parquet (u ONLINE con SQL).

Chat de análisis (Azure OpenAI)

Se construye un system prompt con:

micromomento seleccionado,

histórico de improvements (como JSON),

reglas de no ofrecer “usuarios inspiradores” a menos que el usuario lo pida.

Devuelve resumen y hasta 5 ideas originales (no repite literal el histórico).

Lista de BUs predefinida

BGLA, BUT, CAREPLUS, CHILEINSURANCE, CHILEPROVISION, DENTAL, HOSPITALES, LUXMED, MAYORES, MEXICO, SEGUROS, VITAMEDICA.

Estilos y UX

Botones con ancho uniforme, saltos de línea cuidados y espaciado compacto entre filas.

Historial con burbujas tipo chat y cabecera con icono de descarga.

Solución de problemas

Conexión SQL:

Revisa SQL_DIALECT (pyodbc en local con ODBC Driver 17, o pytds con TLS).

Comprueba SQL_SERVER, SQL_DATABASE, SQL_USERNAME, SQL_PASSWORD.

OFFLINE:

Verifica que los parquet estén en ./data/ o raíz y con las columnas esperadas.

Azure OpenAI:

Revisa AZURE_OPENAI_* y que el deployment exista en tu recurso.
