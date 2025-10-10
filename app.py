import streamlit as st
import os
from dotenv import load_dotenv
import pandas as pd
import certifi
import sqlalchemy as sa
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

import logging
logging.basicConfig()
logging.getLogger("sqlalchemy.engine").setLevel(logging.INFO)

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

# --- Secrets helper: prioriza st.secrets y cae a variables de entorno ---
def cfg(key, default=None):
    try:
        return st.secrets[key]
    except Exception:
        return os.getenv(key, default)

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

def obtener_micromomentos_por_bu(bu, eng):
    try:
        query = f"""
            SELECT DISTINCT micromomento 
            FROM micromomentos_actuar
            WHERE bu = '{bu}'
            ORDER BY micromomento
        """
        df = pd.read_sql(query, eng)
        return df["micromomento"].tolist() if not df.empty else []
    except Exception as e:
        st.warning(f"No se pudieron recuperar micromomentos para {bu}: {e}")
        return []

def obtener_bus_por_micromomento(mm, bu_ref, eng):
    """
    Dado un micromomento 'mm' (texto BU-local) y una BU de referencia 'bu_ref',
    encuentra su micromomento_global y devuelve la lista de BUs donde aparece.
    """
    try:
        subquery = f"""
            SELECT micromomento_global 
            FROM micromomentos_actuar
            WHERE UPPER(bu) = UPPER('{bu_ref}') 
              AND UPPER(micromomento) = UPPER('{mm}')
        """
        df_sub = pd.read_sql(subquery, eng)
        if df_sub.empty:
            return []
        micromomento_global = df_sub.iloc[0]["micromomento_global"]

        query_bu = f"""
            SELECT DISTINCT bu 
            FROM micromomentos_actuar
            WHERE micromomento_global = '{micromomento_global}'
            ORDER BY bu
        """
        df_bu_mm = pd.read_sql(query_bu, eng)
        return df_bu_mm["bu"].tolist() if not df_bu_mm.empty else []
    except Exception as e:
        st.warning(f"No se pudieron recuperar BUs para {mm}: {e}")
        return []

engine = None
if "bu_simulada" in st.session_state:
    try:
        engine = crear_engine()

        # Smoke test de conexión
        with engine.connect() as conn:
            ping = conn.exec_driver_sql("SELECT 1").scalar()
            st.caption(f"Ping SQL: {ping}")
            
        missing = [k for k in ["SQL_SERVER","SQL_DATABASE","SQL_USERNAME","SQL_PASSWORD"] if not cfg(k)]
        if missing:
            st.warning(f"Faltan secretos de BBDD: {', '.join(missing)}")
        st.session_state["micromomentos_simulada"] = obtener_micromomentos_por_bu(st.session_state["bu_simulada"], engine)
    except Exception as e:
        st.error(f"Error al conectar con la base de datos: {e}")

# =========================================================
# 🔹 Interfaz tipo "chat" por BOTONES (sin LLM)
# =========================================================
if "bu_simulada" in st.session_state and engine is not None:

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
        engine_final = crear_engine()
        
        # Smoke test de conexión
        with engine.connect() as conn:
            ping = conn.exec_driver_sql("SELECT 1").scalar()
            st.caption(f"Ping SQL: {ping}")
            
        missing = [k for k in ["SQL_SERVER","SQL_DATABASE","SQL_USERNAME","SQL_PASSWORD"] if not cfg(k)]
        if missing:
            st.warning(f"Faltan secretos de BBDD: {', '.join(missing)}")

        # ============================================
        # Variables de contexto (robustas)
        # ============================================
        mm = (st.session_state.get("mm_seleccionado") or "").strip()
        bu_focus = None
        # Si el usuario acotó explícitamente tras elegir un micromomento:
        if st.session_state.get("bu_mm_seleccionada") and st.session_state["bu_mm_seleccionada"] != "todas":
            bu_focus = st.session_state["bu_mm_seleccionada"]
        # Si el flujo fue al revés (primero BU y luego micromomento), usamos bu_seleccionada:
        elif st.session_state.get("bu_seleccionada"):
            bu_focus = st.session_state["bu_seleccionada"]

        # BU de referencia para mapear micromomento_global cuando se pide "todas las BUs"
        # (usa la que ya venías usando; si no existe, caemos a la simulada o a la de foco)
        bu_ref_global = (
            st.session_state.get("bu_preseleccionada")
            or st.session_state.get("bu_simulada")
            or bu_focus
            or ""
        )

        # Sanitizar comillas simples por seguridad
        def sql_safe(s: str) -> str:
            return s.replace("'", "''") if isinstance(s, str) else s

        mm_safe = sql_safe(mm or "")
        bu_focus_safe = sql_safe(bu_focus or "")
        bu_ref_global_safe = sql_safe(bu_ref_global or "")

        # ============================================
        # Caso 1: Micromomento en todas las BUs
        # ============================================
        if mm and st.session_state.get("bu_mm_seleccionada") == "todas":
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
                    --YEAR(A.FECHA) = 2025
                      --AND MONTH(A.FECHA) BETWEEN 7 AND 9
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

        # ============================================
        # Caso 2: Micromomento para una BU concreta
        # (acepta bu_mm_seleccionada o bu_seleccionada)
        # ============================================
        elif mm and bu_focus:
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
                    --YEAR(A.FECHA) = 2025
                      --AND MONTH(A.FECHA) BETWEEN 7 AND 9
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

        # ============================================
        # Caso 3: Solo BU (todos los micromomentos de esa BU)
        # ============================================
        elif st.session_state.get("bu_seleccionada"):
            bu_only = sql_safe(st.session_state["bu_seleccionada"])
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
                    --YEAR(A.FECHA) = 2025
                      --AND MONTH(A.FECHA) BETWEEN 7 AND 9
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




