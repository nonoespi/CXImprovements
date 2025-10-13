import streamlit as st
# import os
from dotenv import load_dotenv
import pandas as pd
import certifi
import sqlalchemy as sa
from sqlalchemy import text
import urllib
import json
from openai import AzureOpenAI
import time
from io import BytesIO
from datetime import datetime
import base64
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib import colors
from reportlab.lib.units import cm
from pathlib import Path
import tiktoken 

import logging, os
logging.basicConfig()
logging.getLogger("sqlalchemy.engine").setLevel(logging.INFO)  # logs SQL en Cloud

def _mask(v, keep=3):
    if not v: return ""
    v = str(v)
    return v[:keep] + "…" if len(v) > keep else "…"

# 🔹 Siempre limpiar cachés al arrancar
st.cache_data.clear()
st.cache_resource.clear()

# =========================================================
# 🔹 Configuración inicial en la barra lateral
# =========================================================
# Lista fija de BUs
lista_bu = [
    "BGLA", "BUT", "CAREPLUS", "CHILEINSURANCE", "CHILEPROVISION",
    "DENTAL", "HOSPITALES", "LUXMED", "MAYORES", "MEXICO",
    "SEGUROS", "VITAMEDICA"
]

with st.sidebar:
    # Botón para resetear todo
    if st.button("🔄 Resetear demo"):
        st.session_state.clear()
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

    # Selección de BU
    bu_preseleccionada = st.selectbox(
        "Selecciona la BU a simular:",
        lista_bu
    )

    if st.button("Comenzar"):
        st.session_state["bu_simulada"] = bu_preseleccionada

    # Mostrar BU validada
    if "bu_simulada" in st.session_state:
        st.success(f"BU simulada: {st.session_state['bu_simulada']}")

# =========================================================
# 🔹 Carga de variables de entorno y CSS
# =========================================================
load_dotenv()

def cfg(key, default=None):
    # Prioriza st.secrets; si no, variables de entorno (para local)
    try:
        return st.secrets[key]
    except Exception:
        import os
        return os.getenv(key, default)

OFFLINE = cfg("SQL_DIALECT", "pyodbc").lower() == "offline"

@st.cache_data(show_spinner=False)
def _load_parquet(name: str) -> pd.DataFrame:
    base = Path(__file__).parent / "data"
    path = base / name
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}. Sube los .parquet a /data.")
    return pd.read_parquet(path)

@st.cache_data(show_spinner=False)
def load_df_micros() -> pd.DataFrame:
    df = _load_parquet("micromomentos_actuar.parquet")
    df = df.rename(columns={
        "BU": "bu",
        "Micromomento": "micromomento",
        "Micromomento_Global": "micromomento_global",
    })
    return df[["bu", "micromomento", "micromomento_global"]]

@st.cache_data(show_spinner=False)
def load_df_desplegables() -> pd.DataFrame:
    return _load_parquet("datosdesplegables_actuar.parquet")  # BU, Id_Desplegable2/3, Valor_Desplegable2/3

@st.cache_data(show_spinner=False)
def load_df_multiples() -> pd.DataFrame:
    return _load_parquet("datosmultiplesmejorasactuar.parquet")  # ID_MEJORA, Id_Seleccionado

@st.cache_data(show_spinner=False)
def load_df_mejoras() -> pd.DataFrame:
    return _load_parquet("mejorasactuar.parquet")  # ID_MEJORA, BU, ID_USUARIO, FECHA, Titulo, Detalle

@st.cache_data(show_spinner=False)
def load_df_usuarios() -> pd.DataFrame:
    return _load_parquet("usuarios.parquet")  # ID_USUARIO, USUARIO

def count_tokens_messages(messages, encoding_name="cl100k_base"):
    """
    Estima tokens de un payload de chat.messages al estilo OpenAI.
    Fórmula basada en la guía oficial: ~4 tokens por mensaje + contenido.
    """
    try:
        enc = tiktoken.get_encoding(encoding_name)
    except Exception:
        # Fallback robusto si falta la encoding; no debería ocurrir con tiktoken moderno
        enc = tiktoken.get_encoding("cl100k_base")

    tokens = 0
    for m in messages:
        # Cada mensaje aporta ~4 tokens (role, separators…)
        tokens += 4
        for k, v in m.items():
            if isinstance(v, str):
                tokens += len(enc.encode(v))
            # Cuando viene 'name', se resta 1 token
            if k == "name":
                tokens -= 1
    # +2 tokens de primado del asistente
    tokens += 2
    return tokens

st.markdown("""
<style>
.chat-message { padding:10px 15px; border-radius:15px; margin:8px; word-wrap:break-word; }
.chat-message.assistant { text-align:left; margin-right:auto; }
.chat-message.user { background-color:#f7f8fa; text-align:left; margin-left:auto; }

/* === Botones: ancho uniforme === */
.stButton > button {
    width: 100%;
    min-width: 0;
    box-sizing: border-box;
    margin: 4px 6px;
    border-radius: 10px;
    font-size: 0.7rem;
    padding: 10px 12px;
}

/* === IMPORTANTE: NO forzar flex en los bloques horizontales === */
/* (Elimina/No uses reglas sobre div[data-testid="stHorizontalBlock"]) */

/* (Opcional) Asegurar que cada columna ocupe la misma fracción */
div[data-testid="column"] {
    flex: 1 1 0 !important;
    min-width: 0 !important;
}

/* Evitar saltos de línea “feos” en los botones */
.stButton > button {
    white-space: normal;   /* permite romper línea en espacios */
    word-break: keep-all;  /* evita cortar palabras por la mitad */
}

/* === Modo compacto para las filas de botones === */
div[data-testid="column"] .stButton{
    margin-top: 2px !important;      /* antes suele ser ~16px */
    margin-bottom: 2px !important;   /* reduce el hueco entre filas */
}

/* Mantén el botón cómodo pero sin crecer la fila */
.stButton > button{
    padding-top: 4px;
    padding-bottom: 4px;
}

/* (Opcional) quita relleno vertical extra dentro de las columnas */
div[data-testid="column"] > div{
    padding-top: 0 !important;
    padding-bottom: 0 !important;
}
</style>
""", unsafe_allow_html=True)


# ======= CABECERA EN PLACEHOLDER (se rellenará al final) =======
header_ph = st.empty()

st.markdown("""
<style>
.header-flex{display:flex;align-items:center;justify-content:space-between;margin-bottom:1.25rem}
.header-flex h1{font-size:2.1rem;font-weight:700;color:#1E1E1E;margin:0}
.download-btn{display:flex;align-items:center;justify-content:center;
  padding:.4rem;border-radius:8px;text-decoration:none;background:transparent}
.download-btn:hover{background:#F5F5F5}
.dl-ico{width:32px;height:32px;stroke:#444;stroke-width:2.2;
  filter:drop-shadow(0 1px 1px rgba(0,0,0,.1))}
</style>
""", unsafe_allow_html=True)


def _merge_full_chat() -> list:
    full = []
    ch1 = st.session_state.get("chat_history", [])
    ch2 = st.session_state.get("chat_history_analisis", [])
    if ch1:
        full.extend(ch1)
    if ch2:
        full.append({"role": "separator", "content": "--- SEGUNDO CHATBOT (Análisis) ---"})
        full.extend(ch2)
    return full

def _build_pdf_bytes(full_chat: list) -> bytes:
    if not full_chat:
        return b""
    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm
    )
    styles = getSampleStyleSheet()
    user_style = ParagraphStyle("User", parent=styles["Normal"], textColor=colors.black, fontSize=11, leading=14, spaceAfter=8)
    assistant_style = ParagraphStyle("Assistant", parent=styles["Normal"], textColor=colors.HexColor("#0056A3"), fontSize=11, leading=14, spaceAfter=8)
    sep_style = ParagraphStyle("Sep", parent=styles["Normal"], alignment=1, textColor=colors.HexColor("#6c757d"), fontSize=10, spaceBefore=6, spaceAfter=12)
    footer_style = ParagraphStyle("Footer", parent=styles["Normal"], textColor=colors.HexColor("#888"), fontSize=8, spaceBefore=18, alignment=2)

    elements = [Paragraph("<b>Histórico de conversación CX Improvements</b>", styles["Title"]), Spacer(1, 12)]
    for msg in full_chat:
        role = msg.get("role", "")
        content = (msg.get("content", "") or "").replace("\n", "<br/>")
        if role == "separator":
            elements.append(Paragraph(content or "---", sep_style))
        elif role == "assistant":
            elements.append(Paragraph(f"<b>Asistente:</b> {content}", assistant_style))
        elif role == "user":
            elements.append(Paragraph(f"<b>Usuario:</b> {content}", user_style))
        else:
            elements.append(Paragraph(content, user_style))

    ts = datetime.now().strftime("%d/%m/%Y %H:%M")
    elements.append(Paragraph(f"Exportado el {ts}", footer_style))
    doc.build(elements)
    return buf.getvalue()

def update_pdf_bytes():
    """Recalcula y guarda el PDF en sesión. Llamar justo después de cada append al historial."""
    st.session_state["pdf_bytes"] = _build_pdf_bytes(_merge_full_chat())


# =========================================================
# 🔹 Conexión a la BBDD de Azure
# =========================================================
def crear_engine():
    
    if OFFLINE:
        return None  # modo Parquet: no hay SQL
    
    dialect = cfg("SQL_DIALECT", "pyodbc")  # en Streamlit Cloud pon 'pytds' en Secrets

    server   = cfg("SQL_SERVER")
    database = cfg("SQL_DATABASE")
    username = cfg("SQL_USERNAME")
    password = cfg("SQL_PASSWORD")

    if dialect == "pytds":
        # 🔒 TLS activado pasando cafile (certifi)
        return sa.create_engine(
            f"mssql+pytds://{username}:{password}@{server}:1433/{database}?autocommit=true",
            connect_args={"cafile": certifi.where()}  # activa TLS y valida CA
        )

    else:
        # Camino original con ODBC para local (requiere ODBC Driver 17 instalado)
        driver = cfg("SQL_DRIVER", "ODBC Driver 17 for SQL Server")
        connection_string = (
            f"DRIVER={{{driver}}};"
            f"SERVER={server};"
            f"DATABASE={database};"
            f"UID={username};"
            f"PWD={password};"
        )
        params = urllib.parse.quote_plus(connection_string)
        return sa.create_engine(f"mssql+pyodbc:///?odbc_connect={params}")

def probar_conexion(engine):
    """Devuelve (ok: bool, msg: str)."""
    try:
        with engine.connect() as conn:
            _ = conn.exec_driver_sql("SELECT 1").scalar()
        return True, "SELECT 1 OK"
    except Exception as e:
        import traceback
        tb = ''.join(traceback.format_exc())
        return False, f"{repr(e)}\n{tb[-1200:]}"

def obtener_micromomentos_por_bu(bu, eng):
    bu_norm = str(bu).strip().upper()
    if eng is None:  # OFFLINE
        df = load_df_micros()
        mask = df["bu"].astype(str).str.strip().str.upper() == bu_norm
        vals = (df.loc[mask, "micromomento"]
                  .dropna().astype(str).str.strip().unique())
        return sorted(vals.tolist())
    else:  # SQL
        sql = text("""
            SELECT DISTINCT micromomento
            FROM dbo.Micromomentos_Actuar
            WHERE UPPER(LTRIM(RTRIM(bu))) = UPPER(LTRIM(RTRIM(:bu)))
            ORDER BY micromomento
        """)
        df_sql = pd.read_sql_query(sql, eng, params={"bu": bu})
        return df_sql["micromomento"].tolist() if not df_sql.empty else []

def obtener_bus_por_micromomento(mm, bu_ref, eng):
    if eng is None:  # OFFLINE
        df = load_df_micros()
        bu_norm = str(bu_ref).strip().upper()
        mm_norm = str(mm).strip().upper()
        sub = df[
            (df["bu"].astype(str).str.strip().str.upper() == bu_norm) &
            (df["micromomento"].astype(str).str.strip().str.upper() == mm_norm)
        ].head(1)
        if sub.empty:
            return []
        mmg = sub.iloc[0]["micromomento_global"]
        vals = (df.loc[df["micromomento_global"] == mmg, "bu"]
                  .dropna().astype(str).str.strip().unique())
        return sorted(vals.tolist())
    else:  # SQL
        sql1 = text("""
            SELECT TOP 1 micromomento_global
            FROM dbo.Micromomentos_Actuar
            WHERE UPPER(LTRIM(RTRIM(bu))) = UPPER(LTRIM(RTRIM(:bu)))
              AND UPPER(LTRIM(RTRIM(micromomento))) = UPPER(LTRIM(RTRIM(:mm)))
        """)
        df_sub = pd.read_sql_query(sql1, eng, params={"bu": bu_ref, "mm": mm})
        if df_sub.empty:
            return []
        mmg = df_sub.iloc[0]["micromomento_global"]
        sql2 = text("""
            SELECT DISTINCT bu
            FROM dbo.Micromomentos_Actuar
            WHERE micromomento_global = :mmg
            ORDER BY bu
        """)
        df_bu = pd.read_sql_query(sql2, eng, params={"mmg": mmg})
        return df_bu["bu"].tolist() if not df_bu.empty else []

@st.cache_data(show_spinner=False)
def obtener_improvements_offline(
    bu: str | None,
    micromomento: str | None,
    bu_ref_global: str | None = None,   # 👈 NUEVO: BU de referencia para resolver micromomento_global
) -> pd.DataFrame:
    """
    Replica la lógica SQL:
      - Join MEJORASACTUAR + DATOSMULTIPLESMEJORASACTUAR + DATOSDESPLEGABLES_ACTUAR
      - Selección de Id_Desplegable2/3 y Valor_Desplegable2/3 según BU especial ('HOSPITALES','DENTAL','MAYORES')
      - Filtro temporal (último año)
      - Si hay micromomento (mm) sin BU => mapear a micromomento_global usando BU de referencia y traer TODAS las BUs del mismo grupo global
      - Si mm + BU => igual, pero acotando a esa BU
      - Si solo BU => solo filtra por BU
    """
    # --- Cargar datasets ---
    mejoras = load_df_mejoras().copy()
    multiples = load_df_multiples().copy()
    despl    = load_df_desplegables().copy()
    micros   = load_df_micros().copy()  # bu, micromomento, micromomento_global

    # Normalizaciones
    if "BU" in mejoras.columns:
        mejoras["BU"] = mejoras["BU"].astype(str).str.strip().str.upper()
    if "BU" in despl.columns:
        despl["BU"] = despl["BU"].astype(str).str.strip().str.upper()
    micros["bu"] = micros["bu"].astype(str).str.strip().str.upper()
    micros["micromomento"] = micros["micromomento"].astype(str).str.strip()
    micros["micromomento_global"] = micros["micromomento_global"].astype(str).str.strip()

    # --- Elegir para cada BU su Id_Desplegable y valor de micromomento (A2) ---
    especiales = {"HOSPITALES", "DENTAL", "MAYORES"}
    mask_esp = despl["BU"].isin(especiales)
    A2 = pd.DataFrame({
        "BU": despl["BU"],
        "Id_Desplegable": despl["Id_Desplegable2"].where(~mask_esp, despl["Id_Desplegable3"]),
        "Micromomento":   despl["Valor_Desplegable2"].where(~mask_esp, despl["Valor_Desplegable3"]),
    })

    # --- Mejoras + múltiples (A1) ---
    A1 = mejoras.merge(multiples, on="ID_MEJORA", how="left")

    # --- Join por BU y Id_Seleccionado ↔ Id_Desplegable (A) ---
    A = A1.merge(A2, left_on=["BU", "Id_Seleccionado"],
                    right_on=["BU", "Id_Desplegable"], how="left")

    # --- Filtro temporal (último año), igual que en SQL ---
    # En tus queries: CAST(A.FECHA AS DATE) >= DATEADD(YEAR,-1,GETDATE())
    if "FECHA" in A.columns:
        A["FECHA"] = pd.to_datetime(A["FECHA"], errors="coerce")
        from datetime import datetime, timedelta
        cutoff = pd.Timestamp(datetime.now() - timedelta(days=365)).normalize()
        A = A[(A["FECHA"].isna()) | (A["FECHA"].dt.date >= cutoff.date())]

    # --- Resolver filtros en función de mm/bu ---
    # Caso 1 y 2: hay micromomento => mapear a micromomento_global usando bu_ref_global
    if micromomento:
        mm_norm = str(micromomento).strip().upper()
        # BU de referencia: simulada si no se pasa explícita
        bu_ref = (bu_ref_global or st.session_state.get("bu_simulada", "") or "").strip().upper()

        # Buscar micromomento_global de referencia
        sub_ref = micros[
            (micros["bu"] == bu_ref) &
            (micros["micromomento"].str.upper() == mm_norm)
        ].head(1)
        if not sub_ref.empty:
            mmg_ref = sub_ref.iloc[0]["micromomento_global"]
            mmg_ref_norm = str(mmg_ref).strip().upper()

            # Todas las (bu, micromomento) cuyo micromomento_global LIKE mmg_ref
            micros_like = micros[
                micros["micromomento_global"].str.upper().str.contains(mmg_ref_norm, na=False)
            ]
            # Lista de micromomentos “locales” equivalentes
            mms_equivalentes = set(micros_like["micromomento"].astype(str).str.strip())

            # Filtramos A por esos micromomentos equivalentes
            A = A[A["Micromomento"].astype(str).str.strip().isin(mms_equivalentes)]

        else:
            # Si no encontramos mapping, caemos al filtro exacto (mejor que nada)
            A = A[A["Micromomento"].astype(str).str.strip().str.upper() == mm_norm]

    # Filtro por BU (si procede)
    if bu:
        bu_norm = str(bu).strip().upper()
        A = A[A["BU"].astype(str).str.strip().str.upper() == bu_norm]

    # --- Enriquecer con USUARIO (lower) si está disponible ---
    try:
        usuarios = load_df_usuarios()
        A = A.merge(usuarios, on="ID_USUARIO", how="left")
        if "USUARIO" in A.columns:
            A["USUARIO"] = A["USUARIO"].astype(str).str.lower()
    except Exception:
        pass

    # --- Quitar mejoras sin detalle (como en SQL) ---
    if "Detalle" in A.columns:
        A = A[A["Detalle"].notna()]

    # --- Selección/orden final ---
    cols_out = [c for c in ["ID_MEJORA","BU","USUARIO","FECHA","Titulo","Detalle","Micromomento"] if c in A.columns]
    A = A[cols_out].sort_values(
        by=[c for c in ["FECHA","ID_MEJORA"] if c in A.columns],
        ascending=False, na_position="last"
    ).reset_index(drop=True)

    return A


def _resolver_filtros_desde_estado():
    """
    Devuelve (bu_filter, mm_filter) según los estados:
      - Caso 1: micromomento para TODAS las BUs  -> (None, mm)
      - Caso 2: micromomento para una BU concreta -> (bu_mm, mm)
      - Caso 3: solo BU (sin micromomento)       -> (bu, None)
    Soporta valores ausentes y normaliza "todas".
    """
    mm = st.session_state.get("mm_seleccionado")
    bu = st.session_state.get("bu_seleccionada")
    bu_mm = st.session_state.get("bu_mm_seleccionada")  # puede ser "todas"

    # Si viene del flujo de micromomento (tiene bu_mm_seleccionada)
    if mm and bu_mm:
        if isinstance(bu_mm, str) and bu_mm.strip().lower() == "todas":
            return None, mm                   # Caso 1
        else:
            return bu_mm, mm                  # Caso 2

    # Micromomento sin acotar a BU (por si no pasaste por la pregunta de acotación)
    if mm and not bu_mm:
        return None, mm                       # Caso 1 (implícito)

    # Solo BU elegida, sin micromomento
    if bu and not mm:
        return bu, None                       # Caso 3

    # Si por cualquier razón hay ambos pero sin bu_mm, prioriza el BU explícito
    if bu and mm:
        return bu, mm

    # Nada aún seleccionado
    return None, None


# ====== Preparación de datos tras elegir BU simulada (SQL u OFFLINE) ======
engine = crear_engine()
# st.caption("Modo datos: OFFLINE (CSV/Parquet)" if engine is None else "Modo datos: SQL")

if "bu_simulada" in st.session_state:
    bu_sim = st.session_state["bu_simulada"]
    try:
        if engine is not None:
            # --- Rama SQL ---
            ok, msg = probar_conexion(engine)
            if not ok:
                st.error(f"Error al conectar con la base de datos:\n{msg}")
                with st.expander("Diagnóstico SQL (temporal)"):
                    st.write("Dialecto:", cfg("SQL_DIALECT", "pyodbc"))
                    st.write("Servidor:", _mask(cfg("SQL_SERVER")))
                    st.write("BD:", cfg("SQL_DATABASE"))
                    st.write("Usuario:", _mask(cfg("SQL_USERNAME")))
                st.stop()

            # Ping + probe (solo SQL)
            with engine.connect() as conn:
                ping = conn.exec_driver_sql("SELECT 1").scalar()
                st.caption(f"Ping SQL (final): {ping}")
            with engine.connect() as conn:
                df_probe = pd.read_sql_query("SELECT TOP 5 * FROM dbo.Micromomentos_Actuar", conn)
            st.caption(f"Probe Micromomentos_Actuar: filas={len(df_probe)} cols={list(df_probe.columns)}")

            # Lista de micromomentos para la BU simulada (SQL)
            st.session_state["micromomentos_simulada"] = obtener_micromomentos_por_bu(bu_sim, engine)

        else:
            # --- Rama OFFLINE (Parquet) ---
            st.session_state["micromomentos_simulada"] = obtener_micromomentos_por_bu(bu_sim, None)

        # Aviso si faltan secretos cuando estás en SQL
        if engine is not None:
            missing = [k for k in ["SQL_SERVER","SQL_DATABASE","SQL_USERNAME","SQL_PASSWORD"] if not cfg(k)]
            if missing:
                st.warning(f"Faltan secretos de BBDD: {', '.join(missing)}")

    except Exception as e:
        import traceback
        tb = ''.join(traceback.format_exc())
        st.error(f"Error preparando datos: {repr(e)}")
        with st.expander("Diagnóstico SQL (temporal)"):
            st.write("Dialecto:", cfg("SQL_DIALECT", "pyodbc"))
            st.write("Servidor:", _mask(cfg("SQL_SERVER")))
            st.write("BD:", cfg("SQL_DATABASE"))
            st.write("Usuario:", _mask(cfg("SQL_USERNAME")))
            st.code(tb[-2000:])


# =========================================================
# 🔹 Interfaz tipo "chat" por BOTONES (sin LLM)
# =========================================================
if "bu_simulada" in st.session_state:   # ✅ también en OFFLINE

    # Inicializar estados
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("finalizado", False)
    st.session_state.setdefault("fase", None)  # None -> inicio (bloques 1 y 2). Luego: 'micros_por_bu' o 'bus_por_mm'
    st.session_state.setdefault("bu_seleccionada", None)
    st.session_state.setdefault("mm_seleccionado", None)
    st.session_state.setdefault("bu_mm_seleccionada", None)

    # Mensaje de bienvenida (una vez)
    if not st.session_state["chat_history"]:
        st.session_state["chat_history"].append(
            {"role": "assistant",
             "content": "Hola. Soy el asistente de CX de Bupa. Mi objetivo es ayudarte a encontrar inspiración basada en el histórico de Improvements realizadas. ¿Sobre qué te gustaría que profundicemos?"}
        )

    # Render del historial
    for msg in st.session_state["chat_history"]:
        role_class = "assistant" if msg["role"] == "assistant" else "user"
        st.markdown(f'<div class="chat-message {role_class}">{msg["content"]}</div>', unsafe_allow_html=True)

    # ---------------------------
    # Bloques iniciales 1 y 2 (solo al inicio)
    # ---------------------------
    if st.session_state["fase"] is None and not st.session_state.get("finalizado", False):
        # Bloque 1: BUs
        st.markdown('<div class="chat-message assistant">Si lo deseas, podemos profundizar en una BU:</div>', unsafe_allow_html=True)
        cols = st.columns(4)
        for i, bu in enumerate(lista_bu):
            with cols[i % 4]:
                if st.button(bu, key=f"btn_bu_{bu}", use_container_width=True):
                    st.session_state["bu_seleccionada"] = bu
                    st.session_state["chat_history"].append({"role": "user", "content": f"BU seleccionada: {bu}"})
                    # Pasamos a BLOQUE 3 (micromomentos de esa BU) y ocultamos los bloques iniciales
                    st.session_state["micros_por_bu"] = obtener_micromomentos_por_bu(bu, engine)
                    st.session_state["chat_history"].append(
                        {"role": "assistant",
                         "content": f"Estos son los micromomentos disponibles en la BU {bu}. "
                                    f"¿Sobre cuál querrías que nos centráramos?"}
                    )
                    st.session_state["fase"] = "micros_por_bu"
                    update_pdf_bytes()
                    st.rerun()

        # Bloque 2: Micromomentos de la BU simulada
        st.markdown(
            f'<div class="chat-message assistant">Por otro lado, podemos indagar en alguno de los micromomentos de la BU {st.session_state["bu_simulada"]}:</div>',
            unsafe_allow_html=True
        )
        cols2 = st.columns(4)
        for i, mm in enumerate(st.session_state.get("micromomentos_simulada", [])):
            with cols2[i % 4]:
                if st.button(mm, key=f"btn_mm_sim_{mm}", use_container_width=True):
                    st.session_state["mm_seleccionado"] = mm
                    st.session_state["chat_history"].append(
                        {"role": "user", "content": f"Micromomento seleccionado: {mm}"}
                    )
                    # Pasamos a BLOQUE 4 (BUs que tienen ese micromomento) y ocultamos los bloques iniciales
                    bus_mm = obtener_bus_por_micromomento(mm, st.session_state["bu_simulada"], engine)
                    st.session_state["bus_por_mm"] = bus_mm
                    if len(bus_mm) == 1:
                        # Auto-confirmación si solo hay una BU posible
                        unica_bu = bus_mm[0]
                        st.session_state["bu_mm_seleccionada"] = unica_bu
                        st.session_state["chat_history"].append(
                            {"role": "assistant",
                             "content": f"Este micromomento solo está presente en ésta BU. "
                                        f"Vamos a buscar inspiración del micromomento {mm} para la BU {unica_bu}."}
                        )
                        st.session_state["chat_history"].append(
                            {"role": "assistant",
                             "content": f"Recopilando histórico de Improvements. Iniciando chat..."}
                        )
                        st.session_state["finalizado"] = True
                        st.rerun()
                    else:
                        st.session_state["chat_history"].append(
                            {"role": "assistant",
                             "content": f"El micromomento {mm} está presente en las siguientes BUs. "
                                        f"¿Quieres centrarte en una concreta o en todas?"}
                        )
                        st.session_state["fase"] = "bus_por_mm"
                        update_pdf_bytes()
                        st.rerun()

    # ---------------------------
    # Bloque 3: Micromomentos de la BU seleccionada (finaliza al elegir o con 'TODOS'?
    # ---------------------------
    if st.session_state.get("fase") == "micros_por_bu":
        micromomentos = st.session_state.get("micros_por_bu", [])
        # st.markdown(
        #     f'<div class="chat-message assistant">Micromomentos de {st.session_state["bu_seleccionada"]}:</div>',
        #     unsafe_allow_html=True
        # )
        cols3 = st.columns(4)
        opciones = micromomentos + ["TODOS"]
        for i, mm in enumerate(opciones):
            with cols3[i % 4]:
                if st.button(mm, key=f"btn_mm_bu_{mm}", use_container_width=True):
                    if mm == "TODOS":
                        st.session_state["chat_history"].append(
                            {"role": "user", "content": "Micromomento seleccionado: TODOS"}
                        )
                        st.session_state["bu_mm_seleccionada"] = st.session_state["bu_seleccionada"]
                        # time.sleep(0.5)
                        st.session_state["chat_history"].append(
                            {"role": "assistant",
                             "content": f"Vamos a buscar inspiración de la BU {st.session_state['bu_seleccionada']}."}
                        )
                    else:
                        st.session_state["chat_history"].append(
                            {"role": "user", "content": f"Micromomento seleccionado: {mm}"}
                        )
                        st.session_state["mm_seleccionado"] = mm
                        st.session_state["bu_mm_seleccionada"] = st.session_state["bu_seleccionada"]
                        # time.sleep(0.5)
                        st.session_state["chat_history"].append(
                            {"role": "assistant",
                             "content": f"Vamos a buscar inspiración del micromomento {mm} "
                                        f"para la BU {st.session_state['bu_seleccionada']}."}
                        )
                    st.session_state["chat_history"].append(
                            {"role": "assistant",
                             "content": f"Recopilando histórico de Improvements. Iniciando chat..."}
                        )
                    st.session_state["finalizado"] = True
                    st.session_state["fase"] = None  # ocultar botones tras la selección
                    for key in list(st.session_state.keys()):
                        if key.startswith("btn_"):
                            del st.session_state[key]
                    update_pdf_bytes()
                    st.rerun()

    # ---------------------------
    # Bloque 4: BUs con el micromomento seleccionado (incluye 'TODAS') (finaliza al elegir)
    # ---------------------------
    if st.session_state.get("fase") == "bus_por_mm":
        bus_mm = st.session_state.get("bus_por_mm", [])
        # st.markdown(
        #     f'<div class="chat-message assistant">BUs con el micromomento {st.session_state["mm_seleccionado"]}:</div>',
        #     unsafe_allow_html=True
        # )
        cols4 = st.columns(4)
        opciones = bus_mm + ["TODAS"]
        for i, bu in enumerate(opciones):
            with cols4[i % 4]:
                if st.button(bu, key=f"btn_bu_mm_{bu}", use_container_width=True):
                    if bu == "TODAS":
                        st.session_state["chat_history"].append(
                            {"role": "user", "content": "BU seleccionada: TODAS"}
                        )
                        st.session_state["bu_mm_seleccionada"] = "todas"
                        st.session_state["chat_history"].append(
                            {"role": "assistant",
                             "content": f"Vamos a buscar inspiración del micromomento "
                                        f"{st.session_state['mm_seleccionado']} para todas las BUs."}
                        )
                    else:
                        st.session_state["chat_history"].append(
                            {"role": "user", "content": f"BU seleccionada: {bu}"}
                        )
                        st.session_state["bu_mm_seleccionada"] = bu
                        st.session_state["chat_history"].append(
                            {"role": "assistant",
                             "content": f"Vamos a buscar inspiración del micromomento "
                                        f"{st.session_state['mm_seleccionado']} para la BU {bu}."}
                        )
                    st.session_state["chat_history"].append(
                            {"role": "assistant",
                             "content": f"Recopilando histórico de Improvements. Iniciando chat..."}
                        )   
                    st.session_state["finalizado"] = True
                    st.session_state["fase"] = None  # ocultar botones tras la selección
                    for key in list(st.session_state.keys()):
                        if key.startswith("btn_"):
                            del st.session_state[key]
                    update_pdf_bytes()
                    st.rerun()

# =========================================================
# 🔹 SEGUNDO CHATBOT: ANÁLISIS DE HISTÓRICO Y PROPUESTAS
# =========================================================

# =========================================================
# 🔹 Render del segundo chatbot
# =========================================================
if st.session_state.get("finalizado", False):
    try:
        # Resuelve filtros de forma unificada (los 3 casos)
        bu_filter, mm_filter = _resolver_filtros_desde_estado()

        if OFFLINE:
            # ---------- OFFLINE (Parquet) ----------
            df = obtener_improvements_offline(
                bu=bu_filter,
                micromomento=mm_filter
            )

            # 1) Elimina mejoras sin detalle (como ya hacías en SQL)
            if "Detalle" in df.columns:
                df = df[df["Detalle"].notna()]
        
            # 2) Define la clave de “unicidad” de la mejora
            if "ID_MEJORA" in df.columns:
                key_cols = ["ID_MEJORA"]
            else:
                # Fallback si no hubiera ID: aproximamos por BU+Titulo+Detalle
                key_cols = ["BU", "Titulo", "Detalle"]
        
            # 3) Cuenta únicas y, si hay >100, toma muestra aleatoria de *IDs únicos*
            df_unique = df.drop_duplicates(subset=key_cols)
            total_unique = len(df_unique)
        
            if total_unique > 75:
                # Semilla estable por selección (opcional): cambia a None si la quieres cambiar cada rerun
                # from hashlib import sha256
                # seed = int(sha256(f"{bu_filter}|{mm_filter}".encode()).hexdigest()[:8], 16)
                seed = None
        
                sampled_keys = (
                    df_unique
                    .sample(n=100, random_state=seed)
                    [key_cols]
                )
        
                # 4) Filtra el df original a esas mejoras únicas seleccionadas
                df = df.merge(sampled_keys, on=key_cols, how="inner").reset_index(drop=True)
        
                # 5) Aviso
                st.caption(f"Se han seleccionado aleatoriamente 75 improvements de las {total_unique} disponibles (modo DEMO).")

        else:
            # ---------- SQL ----------
            engine_final = crear_engine()
            st.caption("Modo datos: SQL")

            ok, msg = probar_conexion(engine_final)
            if not ok:
                st.error(f"Error al conectar con la base de datos:\n{msg}")
                with st.expander("Diagnóstico SQL (temporal)"):
                    st.write("Dialecto:", cfg("SQL_DIALECT", "pyodbc"))
                    st.write("Servidor:", _mask(cfg("SQL_SERVER")))
                    st.write("BD:", cfg("SQL_DATABASE"))
                    st.write("Usuario:", _mask(cfg("SQL_USERNAME")))
                st.stop()

            # (Opcional) Ping
            with engine_final.connect() as conn:
                st.caption(f"Ping SQL (final): {conn.exec_driver_sql('SELECT 1').scalar()}")

            # Sanitizar comillas para el SQL dinámico que ya tienes
            def sql_safe(s: str) -> str:
                return s.replace("'", "''") if isinstance(s, str) else s

            mm = mm_filter or ""
            bu_focus = bu_filter or ""
            bu_ref_global = (
                st.session_state.get("bu_preseleccionada")
                or st.session_state.get("bu_simulada")
                or bu_focus
                or ""
            )

            mm_safe = sql_safe(mm)
            bu_focus_safe = sql_safe(bu_focus)
            bu_ref_global_safe = sql_safe(bu_ref_global)

            if mm and not bu_focus:
                # === Caso 1: micromomento en TODAS las BUs ===
                query = f"""
                DECLARE @MM_BU NVARCHAR(200), @BU NVARCHAR(200), @MM NVARCHAR(200), @MM_LIKE NVARCHAR(200);
                SET @MM_BU='{mm_safe}';
                SET @BU='{bu_ref_global_safe}';
                SET @MM =(SELECT micromomento_global FROM micromomentos_actuar WHERE bu=@BU AND micromomento=@MM_BU);
                SET @MM_LIKE ='%' +@MM+ '%';

                SELECT @MM_BU AS micromomento, A.BU, LOWER(C.USUARIO) AS USUARIO, A.TITULO+': '+A.DETALLE AS MEJORA
                FROM (
                    SELECT A1.*, A2.Micromomento
                    FROM (
                        SELECT A.ID_MEJORA, UPPER(A.BU) AS BU, A.ID_USUARIO, A.FECHA, A.Titulo, A.Detalle,
                               B.Id_Seleccionado AS Id_Desplegable
                        FROM MEJORASACTUAR A
                        LEFT JOIN DATOSMULTIPLESMEJORASACTUAR B ON A.ID_MEJORA = B.ID_MEJORA
                        WHERE CAST(A.FECHA AS DATE)>=CAST(DATEADD(YEAR,-1,GETDATE()) AS DATE)
                          AND (
                              (B.Id_Desplegable = 'Id_Desplegable3' AND A.BU IN ('HOSPITALES', 'DENTAL', 'MAYORES'))
                              OR
                              (B.Id_Desplegable = 'Id_Desplegable2' AND A.BU NOT IN ('HOSPITALES', 'DENTAL', 'MAYORES'))
                          )
                    ) A1
                    LEFT JOIN (
                        SELECT DISTINCT BU,
                               CASE WHEN BU IN ('HOSPITALES', 'DENTAL', 'MAYORES') THEN Id_Desplegable3 ELSE Id_Desplegable2 END AS Id_Desplegable,
                               CASE WHEN BU IN ('HOSPITALES', 'DENTAL', 'MAYORES') THEN Valor_Desplegable3 ELSE Valor_Desplegable2 END AS Micromomento
                        FROM DATOSDESPLEGABLES_ACTUAR
                    ) A2 ON A1.BU = A2.BU AND A1.Id_Desplegable = A2.Id_Desplegable
                ) A
                RIGHT JOIN (SELECT * FROM Micromomentos_Actuar WHERE micromomento_global LIKE @MM_LIKE) B
                    ON A.BU=B.BU AND A.MICROMOMENTO=B.MICROMOMENTO
                RIGHT JOIN (SELECT ID_USUARIO, USUARIO FROM USUARIOS) C
                    ON A.ID_USUARIO=C.ID_USUARIO
                WHERE A.DETALLE IS NOT NULL
                ORDER BY A.ID_MEJORA DESC;
                """
                df = pd.read_sql(query, engine_final)

            elif mm and bu_focus:
                # === Caso 2: micromomento + BU concreta ===
                query = f"""
                DECLARE @MM_BU NVARCHAR(200), @BU NVARCHAR(200), @MM NVARCHAR(200), @MM_LIKE NVARCHAR(200);
                SET @MM_BU='{mm_safe}';
                SET @BU='{bu_focus_safe}';
                SET @MM =(SELECT micromomento_global FROM micromomentos_actuar WHERE bu=@BU AND micromomento=@MM_BU);
                SET @MM_LIKE ='%' +@MM+ '%';

                SELECT @MM_BU AS micromomento, A.BU, LOWER(C.USUARIO) AS USUARIO, A.TITULO+': '+A.DETALLE AS MEJORA
                FROM (
                    SELECT A1.*, A2.Micromomento
                    FROM (
                        SELECT A.ID_MEJORA, UPPER(A.BU) AS BU, A.ID_USUARIO, A.FECHA, A.Titulo, A.Detalle,
                               B.Id_Seleccionado AS Id_Desplegable
                        FROM MEJORASACTUAR A
                        LEFT JOIN DATOSMULTIPLESMEJORASACTUAR B ON A.ID_MEJORA = B.ID_MEJORA
                        WHERE CAST(A.FECHA AS DATE)>=CAST(DATEADD(YEAR,-1,GETDATE()) AS DATE)
                          AND A.BU=@BU
                          AND (
                              (B.Id_Desplegable = 'Id_Desplegable3' AND A.BU IN ('HOSPITALES', 'DENTAL', 'MAYORES'))
                              OR
                              (B.Id_Desplegable = 'Id_Desplegable2' AND A.BU NOT IN ('HOSPITALES', 'DENTAL', 'MAYORES'))
                          )
                    ) A1
                    LEFT JOIN (
                        SELECT DISTINCT BU,
                               CASE WHEN BU IN ('HOSPITALES', 'DENTAL', 'MAYORES') THEN Id_Desplegable3 ELSE Id_Desplegable2 END AS Id_Desplegable,
                               CASE WHEN BU IN ('HOSPITALES', 'DENTAL', 'MAYORES') THEN Valor_Desplegable3 ELSE Valor_Desplegable2 END AS Micromomento
                        FROM DATOSDESPLEGABLES_ACTUAR
                    ) A2 ON A1.BU = A2.BU AND A1.Id_Desplegable = A2.Id_Desplegable
                ) A
                RIGHT JOIN (SELECT * FROM Micromomentos_Actuar WHERE micromomento_global LIKE @MM_LIKE) B
                    ON A.BU=B.BU AND A.MICROMOMENTO=B.MICROMOMENTO
                RIGHT JOIN (SELECT ID_USUARIO, USUARIO FROM USUARIOS) C
                    ON A.ID_USUARIO=C.ID_USUARIO
                WHERE A.DETALLE IS NOT NULL
                ORDER BY A.ID_MEJORA DESC;
                """
                df = pd.read_sql(query, engine_final)

            elif bu_focus and not mm:
                # === Caso 3: solo BU ===
                bu_only = sql_safe(bu_focus)
                query = f"""
                DECLARE @BU NVARCHAR(200);
                SET @BU='{bu_only}';

                SELECT micromomento, A.BU, LOWER(C.USUARIO) AS USUARIO, A.TITULO+': '+A.DETALLE AS MEJORA
                FROM (
                    SELECT A1.*, A2.Micromomento
                    FROM (
                        SELECT A.ID_MEJORA, UPPER(A.BU) AS BU, A.ID_USUARIO, A.FECHA, A.Titulo, A.Detalle,
                               B.Id_Seleccionado AS Id_Desplegable
                        FROM MEJORASACTUAR A
                        LEFT JOIN DATOSMULTIPLESMEJORASACTUAR B ON A.ID_MEJORA = B.ID_MEJORA
                        WHERE CAST(A.FECHA AS DATE)>=CAST(DATEADD(YEAR,-1,GETDATE()) AS DATE)
                          AND A.BU=@BU
                          AND (
                              (B.Id_Desplegable = 'Id_Desplegable3' AND A.BU IN ('HOSPITALES', 'DENTAL', 'MAYORES'))
                              OR
                              (B.Id_Desplegable = 'Id_Desplegable2' AND A.BU NOT IN ('HOSPITALES', 'DENTAL', 'MAYORES'))
                          )
                    ) A1
                    LEFT JOIN (
                        SELECT DISTINCT BU,
                               CASE WHEN BU IN ('HOSPITALES', 'DENTAL', 'MAYORES') THEN Id_Desplegable3 ELSE Id_Desplegable2 END AS Id_Desplegable,
                               CASE WHEN BU IN ('HOSPITALES', 'DENTAL', 'MAYORES') THEN Valor_Desplegable3 ELSE Valor_Desplegable2 END AS Micromomento
                        FROM DATOSDESPLEGABLES_ACTUAR
                    ) A2 ON A1.BU = A2.BU AND A1.Id_Desplegable = A2.Id_Desplegable
                ) A
                RIGHT JOIN (SELECT ID_USUARIO, USUARIO FROM USUARIOS) C
                    ON A.ID_USUARIO=C.ID_USUARIO
                WHERE A.DETALLE IS NOT NULL
                ORDER BY A.ID_MEJORA DESC;
                """
                df = pd.read_sql(query, engine_final)

            else:
                df = pd.DataFrame()

        if df.empty:
            st.info("No se encontraron Improvements para esta selección.")

    except Exception as e:
        st.error(f"Error al recuperar Improvements: {e}")

    # =========================================================
    # 🔹 Segundo chatbot: análisis del histórico y propuestas
    # =========================================================
    if "analisis_iniciado" not in st.session_state:
        # Guardamos el histórico para el segundo chatbot
        if 'df' in locals() and not df.empty:
            st.session_state["historico_mejoras"] = df.to_dict(orient="records")
        else:
            st.session_state["historico_mejoras"] = []

        st.session_state["chat_history_analisis"] = [
            {"role": "assistant", "content": "He recopilado el histórico de Improvements. ¿Quieres que te muestre un resumen y algunas Improvements inspiradas?"}
        ]
        update_pdf_bytes()
        st.session_state["analisis_iniciado"] = True
        st.rerun()

    # =========================================================
    # 🔹 Render del chatbot de análisis (versión nativa Streamlit)
    # =========================================================
    if "chat_history_analisis" not in st.session_state:
        st.session_state["chat_history_analisis"] = [
            {"role": "assistant", "content": "He recopilado el histórico de Improvements. ¿Quieres que te muestre un resumen y algunas Improvements inspiradas?"}
        ]
        update_pdf_bytes()

    # Mostrar historial de mensajes con API nativa
    for msg in st.session_state["chat_history_analisis"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # =========================================================
    # 🔹 Entrada del usuario
    # =========================================================
    if prompt := st.chat_input("Escribe tu mensaje..."):
        st.session_state["chat_history_analisis"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # ---- Llamada a Azure OpenAI ----
        client = AzureOpenAI(
            api_key=cfg("AZURE_OPENAI_API_KEY"),
            api_version=cfg("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=cfg("AZURE_OPENAI_ENDPOINT"),
        )
        deployment = cfg("AZURE_OPENAI_DEPLOYMENT")

        micromomento = st.session_state.get("mm_seleccionado") or "N/A"
        historico = st.session_state.get("historico_mejoras", [])

        system_prompt = f"""
        Eres un asesor experto de Bupa, referente internacional en gestión y optimización de la experiencia de cliente (CX Improvements). Tu función es:

        1. Recepcionar un micromomento seleccionado por el usuario (de una lista dada).
        2. Analizar el histórico completo de Improvements implementadas, que incluye:
           - BU (Business Unit) asociado.
           - Micromomentos impactados (uno o varios por acción).
           - Usuario que propuso cada acción.
        3. Extraer aprendizajes clave de las iniciativas previas.
        4. Generar hasta 5 Improvements originales y accionables:
           - No repetir literalmente acciones pasadas.
           - Ser innovador, concreto y adaptado al contexto internacional de Bupa.
           - Para cada sugerencia, indicar el beneficio, público objetivo o enfoque diferencial.
        5. Identificación de usuarios inspiradores:
           - Este paso **solo debe realizarse si el usuario lo solicita explícitamente**.
           - En ningún caso debes mencionarlo, insinuarlo ni ofrecerlo de manera proactiva.
           - Si el usuario lo pide, busca en el histórico acciones similares ya implementadas y muestra, como máximo, 3 usuarios por sugerencia.
           - Para cada usuario, incluye:
             - Correo de contacto
             - BU
             - Breve resumen de la acción previa relacionada
           - Si no hay usuarios relacionados, indícalo con claridad.

        Importante: Nunca menciones ni insinúes la existencia de usuarios inspiradores a menos que el usuario lo pida explícitamente.

        ---
        
        Formato de salida si solicitan resumen del histórico:

        **Resumen breve del histórico**
           - Enumera las principales acciones previas relacionadas con el micromomento seleccionado, desglosadas por BU. Pero nunca menciones el micromomento.
           - Si no hay acciones previas, indícalo claramente y sugiere buenas prácticas generales de CX adaptadas a Bupa.

        Formato de salida si solicitan sugerencias inspiradoras y originales:

        **Sugerencias de nuevas improvements**
           - Cada sugerencia debe incluir:
             - **Título breve**
             - **Descripción** (beneficio, público objetivo o enfoque diferencial)
             -
           - No repetir literalmente acciones anteriores. Combinar, evolucionar o adaptar ideas para aportar valor añadido.

        Formato de salida si solicitan usuarios inspiradores:
        
        **Usuarios con improvements similares** *(solo si el usuario lo pide expresamente)*
           - Este bloque debe omitirse por completo salvo que el usuario lo pida.
           - En caso afirmativo, mostrar hasta 3 usuarios por sugerencia (nunca repetir el mismo usuario, aunque tenga varias Improvements relacionadas): 
                - Sugerencia: [Título de la sugerencia] 
                - Usuario 1: [correo de contacto] 
                    BU: [BU] 
                    Improvement relacionada: [breve resumen] 
                    
                - Usuario 2: [...] 
                - Usuario 3: [...] 
                
             Este bloque debe ayudar al usuario a identificar compañeros a quienes consultar si desea desarrollar alguna de las Improvements propuestas.

        ---

        - Mantén un tono directo y profesional, sin informalidades ni conversación secundaria.
        - Usa **markdown simple** (listas, numeración, negritas, cursivas) para estructurar la respuesta. Evita encabezados tipo `###`.

        ---

        Restricción de uso:

        Este modelo está diseñado exclusivamente para:

        - Proporcionar **sugerencias inspiradas y originales** de nuevas Improvements.
        - Facilitar la **identificación de compañeros** que han desarrollado Improvements similares, como fuente de inspiración o contacto (solo si el usuario lo pide).
        - Dar opinión sobre las Improvements, con posibilidad de expresar cuáles son más importantes para mejorar la experiencia de cliente.
        - Dar cualquier tipo de métricas siempre y cuando estén relacionadas con el histórico de Improvements seleccionado (cuántas Improvements hay, usuarios con más Improvements realizadas...).
        - En definitiva, puedes hacer comentarios siempre y cuando esté relacionado con el histórico de Improvements que has recopilado.

        Si el usuario solicita cualquier otro tipo de información no relacionada con este propósito (por ejemplo: datos personales, consultas fuera de contexto, información confidencial no vinculada a Improvements), el modelo debe rechazar educadamente la solicitud y mostrar el siguiente mensaje:

        "Este asistente está diseñado únicamente para facilitar la inspiración en nuevas Improvements y para ayudarte a contactar con compañeros que hayan hecho Improvements similares. No puedo ayudarte con otro tipo de consultas."

        ---

        Micromomento seleccionado: {micromomento}
        Histórico de Improvements (JSON): {json.dumps(historico, ensure_ascii=False)}
        """

        # 👉 NUEVO: construir el payload 'messages' explícitamente y diagnosticar
        messages_payload = [
            {"role": "system", "content": system_prompt},
            *st.session_state["chat_history_analisis"],
            {"role": "user", "content": prompt}
        ]
        
        # Estimar tokens
        try:
            tokens_est = count_tokens_messages(messages_payload)
        except Exception as _tok_err:
            tokens_est = None
        
        # Panel de diagnóstico visible en UI
        with st.expander("🧪 Diagnóstico: payload enviado y tokens (estimado)", expanded=False):
            st.write("Modelo/Deployment:", deployment)
            st.write("API version:", cfg("AZURE_OPENAI_API_VERSION"))
            if tokens_est is not None:
                st.write(f"Tokens estimados del payload: **{tokens_est}**")
            else:
                st.write("No se pudo estimar tokens.")
            # Mostrar una vista del system prompt (truncado por seguridad visual si es largo)
            max_chars = 2000
            sp_preview = system_prompt if len(system_prompt) <= max_chars else system_prompt[:max_chars] + " …[truncado]"
            st.subheader("System prompt (preview)")
            st.code(sp_preview)
            st.subheader("messages (payload completo)")
            st.json(messages_payload)
        
        # 👉 FIN NUEVO

        try:
            response = client.chat.completions.create(
                model=deployment,
                temperature=1,
                messages=[
                    {"role": "system", "content": system_prompt},
                    *st.session_state["chat_history_analisis"],
                    {"role": "user", "content": prompt}
                ]
            )
            answer = response.choices[0].message.content.strip()
        except Exception as e:
            answer = f"Error al contactar con el modelo: {e}"

        # Guardar y mostrar la respuesta
        st.session_state["chat_history_analisis"].append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)
            update_pdf_bytes()

def _merge_full_chat():
    full = []
    ch1 = st.session_state.get("chat_history", [])
    ch2 = st.session_state.get("chat_history_analisis", [])
    if ch1:
        full.extend(ch1)
    if ch2:
        full.extend(ch2)
    return full

def _build_pdf_bytes(full_chat):
    if not full_chat: return b""
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    user_style = ParagraphStyle("User", parent=styles["Normal"], textColor=colors.HexColor("#0056A3"),
                                fontSize=11, leading=14, spaceAfter=8)
    assistant_style = ParagraphStyle("Assistant", parent=styles["Normal"],
                                     textColor=colors.black,
                                     fontSize=11, leading=14, spaceAfter=8)
    sep_style = ParagraphStyle("Sep", parent=styles["Normal"], alignment=1,
                               textColor=colors.HexColor("#6c757d"),
                               fontSize=10, spaceBefore=6, spaceAfter=12)
    footer_style = ParagraphStyle("Footer", parent=styles["Normal"],
                                  textColor=colors.HexColor("#888"),
                                  fontSize=8, spaceBefore=18, alignment=2)

    elements = [Paragraph("<b>CX Improvements</b>", styles["Title"]), Spacer(1, 12)]
    for msg in full_chat:
        role = msg.get("role","")
        content = (msg.get("content","") or "").replace("\n","<br/>")
        if role == "separator":
            elements.append(Paragraph(content or "---", sep_style))
        elif role == "assistant":
            elements.append(Paragraph(f"<b>Asistente:</b> {content}", assistant_style))
        elif role == "user":
            elements.append(Paragraph(f"<b>Usuario:</b> {content}", user_style))
        else:
            elements.append(Paragraph(content, user_style))

    ts = datetime.now().strftime("%d/%m/%Y %H:%M")
    elements.append(Paragraph(f"Exportado el {ts}", footer_style))
    doc.build(elements)
    return buf.getvalue()
    
    
# 1) Fusionar chats con TODO lo que haya ya en sesión (incluye el último turno)
_full = _merge_full_chat()
pdf_bytes = _build_pdf_bytes(_full)

import base64  # si no está arriba
b64 = base64.b64encode(pdf_bytes).decode()

# 2) Pintar la cabecera en el placeholder (el PDF ya está listo)
svg_icon = """
<svg class="dl-ico" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none">
  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
  <path d="M7 10l5 5 5-5"/>
  <path d="M12 15V3"/>
</svg>
"""

with header_ph.container():
    st.markdown(f"""
    <div class="header-flex">
      <h1>CX Improvements</h1>
      {'<a class="download-btn" download="historico_conversacion.pdf" href="data:application/pdf;base64,'+b64+'" title="Descargar PDF">'+svg_icon+'</a>' if pdf_bytes else ''}
    </div>

    """, unsafe_allow_html=True)



