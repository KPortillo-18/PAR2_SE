import re
import unicodedata
from collections import Counter
from urllib.parse import urlparse, parse_qs

import numpy as np
import pandas as pd
import requests
import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud

st.set_page_config(page_title="Análisis de Sentimiento", page_icon="🧠", layout="wide")


# ═══════════════════════════════════════════════════════════
# 1. REGLAS Y PONDERACIONES (aquí puedes ajustar los valores)
# ═══════════════════════════════════════════════════════════

# Adjetivos / palabras con carga emocional. Las palabras que no están
# aquí (comida, servicio, mesero, postre, carro, computadora...) valen 0.
LEXICO_BASE = {
    # positivas
    "excelente": 0.9, "delicioso": 0.8, "deliciosa": 0.8, "rico": 0.7, "rica": 0.7,
    "bueno": 0.7, "buena": 0.7, "rápido": 0.6, "rápida": 0.6, "limpio": 0.6,
    "limpia": 0.6, "amable": 0.7, "genial": 0.8, "increíble": 0.9, "perfecto": 0.9,
    "perfecta": 0.9, "fantástico": 0.9, "maravilloso": 0.9, "espectacular": 0.9,
    "bonito": 0.6, "lindo": 0.6, "hermoso": 0.8, "agradable": 0.6, "mejor": 0.7,
    "bien": 0.5, "divertido": 0.7, "útil": 0.6, "fácil": 0.5, "feliz": 0.7,
    "recomendado": 0.6, "encanta": 0.8, "satisfecho": 0.7,
    # negativas
    "frío": -0.5, "fría": -0.5, "lento": -0.6, "lenta": -0.6, "sucio": -0.6,
    "sucia": -0.6, "grosero": -0.7, "grosera": -0.7, "pésimo": -0.7, "pésima": -0.7,
    "malo": -0.7, "mala": -0.7, "mal": -0.6, "feo": -0.6, "fea": -0.6,
    "horrible": -0.9, "terrible": -0.9, "asqueroso": -0.9, "malísimo": -0.9,
    "peor": -0.8, "odio": -0.9, "lamentable": -0.8, "desagradable": -0.6,
    "decepcionante": -0.7, "aburrido": -0.5, "defectuoso": -0.7, "inútil": -0.7,
    "caro": -0.4, "tardado": -0.5, "difícil": -0.4, "problema": -0.5, "queja": -0.5,
}

MODIFICADORES_BASE = {
    "muy": 1.5, "bastante": 1.5,              # +50 %
    "extremadamente": 2.0, "sumamente": 2.0,  # el doble
}

NEGACIONES = {"no", "nunca", "jamas"}
FACTOR_NEGACION = -0.5     # invierte el signo y reduce la magnitud al 50 %
PESO_C1 = 0.5              # peso del bloque anterior al "pero" / "sin embargo"


def normalizar(texto):
    """Minúsculas y sin tildes, para comparar palabras sin importar la ortografía."""
    texto = unicodedata.normalize("NFD", texto.lower())
    return "".join(c for c in texto if unicodedata.category(c) != "Mn")


# Diccionarios listos para comparar (sin tildes) y para mostrar (con tildes)
LEXICO = {normalizar(p): v for p, v in LEXICO_BASE.items()}
MOSTRAR = {normalizar(p): p for p in LEXICO_BASE}
MODIFICADORES = {normalizar(p): v for p, v in MODIFICADORES_BASE.items()}


# ═══════════════════════════════════════════════════════════
# 2. MOTOR DE ANÁLISIS
# ═══════════════════════════════════════════════════════════

def puntuar_bloque(tokens):
    """Suma la carga emocional de un bloque aplicando modificadores y negación."""
    total, positivas, negativas = 0.0, [], []
    for i, palabra in enumerate(tokens):
        if palabra not in LEXICO:
            continue
        valor = LEXICO[palabra]
        previos = tokens[max(0, i - 3):i]
        for p in previos[-2:]:                      # "muy", "extremadamente"...
            if p in MODIFICADORES:
                valor *= MODIFICADORES[p]
        if any(p in NEGACIONES for p in previos):   # "no", "nunca"
            valor *= FACTOR_NEGACION
        total += valor
        if LEXICO[palabra] > 0:
            positivas.append(MOSTRAR[palabra])
        else:
            negativas.append(MOSTRAR[palabra])
    return total, positivas, negativas


def clasificar(puntaje):
    if puntaje > 1.0:
        return "Positivo Alto"
    if puntaje > 0.2:
        return "Positivo Moderado"
    if puntaje >= -0.2:
        return "Neutral"
    if puntaje >= -1.0:
        return "Negativo Moderado"
    return "Muy Negativo"


def analizar(texto):
    """Devuelve puntaje, nivel y palabras clave de un comentario."""
    # "pero" / "sin embargo" dividen el texto en bloques C1 y C2
    bloques = re.split(r"\b(?:pero|sin embargo)\b", normalizar(texto))
    puntaje, positivas, negativas = 0.0, [], []
    for i, bloque in enumerate(bloques):
        valor, pos, neg = puntuar_bloque(re.findall(r"[a-z]+", bloque))
        puntaje = valor if i == 0 else PESO_C1 * puntaje + valor
        positivas += pos
        negativas += neg
    puntaje = round(puntaje, 3)
    return {"puntaje": puntaje, "nivel": clasificar(puntaje),
            "positivas": positivas, "negativas": negativas}


# ═══════════════════════════════════════════════════════════
# 3. YOUTUBE API
# ═══════════════════════════════════════════════════════════

def obtener_api_key():
    try:
        return st.secrets["YOUTUBE_API_KEY"]
    except Exception:
        return ""


def extraer_video_id(entrada):
    entrada = entrada.strip()
    if re.fullmatch(r"[\w-]{11}", entrada):
        return entrada
    url = urlparse(entrada)
    host = url.hostname or ""
    if "youtu.be" in host:
        return url.path.strip("/")[:11]
    if "youtube.com" in host:
        if url.path == "/watch":
            return parse_qs(url.query).get("v", [None])[0]
        partes = url.path.strip("/").split("/")
        if len(partes) >= 2 and partes[0] in ("shorts", "embed", "live"):
            return partes[1]
    return None


@st.cache_data(ttl=600, show_spinner=False)
def obtener_comentarios_youtube(video_id, api_key, maximo):
    url = "https://www.googleapis.com/youtube/v3/commentThreads"
    comentarios, token = [], None
    while len(comentarios) < maximo:
        params = {
            "part": "snippet",
            "videoId": video_id,
            "maxResults": min(100, maximo - len(comentarios)),
            "textFormat": "plainText",
            "key": api_key,
        }
        if token:
            params["pageToken"] = token
        respuesta = requests.get(url, params=params, timeout=15)
        datos = respuesta.json()
        if respuesta.status_code != 200:
            raise RuntimeError(datos.get("error", {}).get("message", "Error desconocido"))
        for item in datos.get("items", []):
            comentarios.append(item["snippet"]["topLevelComment"]["snippet"]["textDisplay"])
        token = datos.get("nextPageToken")
        if not token:
            break
    return comentarios


# ═══════════════════════════════════════════════════════════
# 4. GRÁFICAS
# ═══════════════════════════════════════════════════════════

def grafica_barras(serie, colores, titulo):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    barras = ax.bar(serie.index, serie.values, color=colores)
    ax.bar_label(barras)
    ax.set_title(titulo)
    ax.set_ylabel("Comentarios")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    return fig


def nube_circular(frecuencias, colormap):
    y, x = np.ogrid[:500, :500]
    mascara = (((x - 250) ** 2 + (y - 250) ** 2) > 240 ** 2).astype(np.uint8) * 255
    nube = WordCloud(width=500, height=500, background_color="white", mask=mascara,
                     colormap=colormap).generate_from_frequencies(frecuencias)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(nube, interpolation="bilinear")
    ax.axis("off")
    return fig


# ═══════════════════════════════════════════════════════════
# 5. INTERFAZ
# ═══════════════════════════════════════════════════════════

COMENTARIOS_INTERNOS = [
    "La comida estaba excelente y el servicio muy rápido",
    "El mesero fue grosero y la comida llegó fría",
    "El postre es extremadamente delicioso",
    "El servicio es sumamente lento y el lugar está sucio",
    "La comida es buena pero el servicio es lento",
    "El restaurante es caro, sin embargo la comida es excelente",
    "No es feo",
    "El carro es nuevo",
    "Pésimo servicio, nunca vuelvo",
    "La computadora es muy buena y bastante rápida",
]

st.title("🧠 Análisis de Sentimiento")
st.caption("Puntaje de -1 a +1 (o más) con reglas de ponderación, escala de 5 niveles y comentarios de YouTube.")

# ── Probar una frase ────────────────────────────────────────
with st.container(border=True):
    frase = st.text_input("Prueba una frase", value="La comida es buena pero el servicio es lento")
    if frase.strip():
        r = analizar(frase)
        c1, c2 = st.columns(2)
        c1.metric("Puntaje", r["puntaje"])
        c2.metric("Nivel", r["nivel"])

st.divider()
st.header("📂 Análisis de un conjunto de comentarios")

fuente = st.radio(
    "¿De dónde vienen los comentarios?",
    ["Comentarios internos de ejemplo", "Subir archivo (.txt / .csv)", "Video de YouTube"],
    horizontal=True,
)

comentarios = []

if fuente == "Comentarios internos de ejemplo":
    comentarios = COMENTARIOS_INTERNOS

elif fuente == "Subir archivo (.txt / .csv)":
    archivo = st.file_uploader(
        "Un comentario por línea (.txt) o una columna de comentarios (.csv)", type=["txt", "csv"]
    )
    if archivo is not None:
        if archivo.name.endswith(".csv"):
            df_archivo = pd.read_csv(archivo)
            comentarios = df_archivo.iloc[:, 0].astype(str).tolist()
        else:
            texto = archivo.read().decode("utf-8")
            comentarios = [l.strip() for l in texto.splitlines() if l.strip()]

else:
    api_key = obtener_api_key() or st.text_input("API key de YouTube", type="password")
    enlace = st.text_input("Enlace del video de YouTube")
    maximo = st.slider("Cantidad de comentarios a traer", 10, 300, 100, step=10)
    if st.button("Obtener comentarios"):
        video_id = extraer_video_id(enlace)
        if not api_key:
            st.error("Falta la API key de YouTube.")
        elif not video_id:
            st.error("No se pudo reconocer el enlace del video.")
        else:
            try:
                with st.spinner("Consultando YouTube..."):
                    st.session_state["comentarios_yt"] = obtener_comentarios_youtube(
                        video_id, api_key, maximo
                    )
            except Exception as e:
                st.error(f"Error de la API de YouTube: {e}")
    comentarios = st.session_state.get("comentarios_yt", [])

if not comentarios:
    st.info("Aún no hay comentarios para analizar.")
    st.stop()

st.success(f"Se analizarán {len(comentarios)} comentarios.")

# ── Analizar todos los comentarios ──────────────────────────
filas, freq_pos, freq_neg = [], Counter(), Counter()
for c in comentarios:
    r = analizar(c)
    freq_pos.update(r["positivas"])
    freq_neg.update(r["negativas"])
    filas.append({"Comentario": c, "Puntaje": r["puntaje"], "Nivel": r["nivel"]})
df = pd.DataFrame(filas)

st.subheader("Resultados")
st.dataframe(df)

# ── Gráficas de conteo ──────────────────────────────────────
st.subheader("📊 Positivos, neutros y negativos")

grupos = df["Nivel"].map({
    "Positivo Alto": "Positivo", "Positivo Moderado": "Positivo",
    "Neutral": "Neutral",
    "Negativo Moderado": "Negativo", "Muy Negativo": "Negativo",
})
conteo_grupos = grupos.value_counts().reindex(["Positivo", "Neutral", "Negativo"], fill_value=0)

niveles = ["Positivo Alto", "Positivo Moderado", "Neutral", "Negativo Moderado", "Muy Negativo"]
conteo_niveles = df["Nivel"].value_counts().reindex(niveles, fill_value=0)

g1, g2 = st.columns(2)
with g1:
    st.pyplot(grafica_barras(conteo_grupos, ["green", "gray", "red"], "Favorables / Neutros / No favorables"))
with g2:
    st.pyplot(grafica_barras(conteo_niveles, ["darkgreen", "limegreen", "gray", "orange", "red"], "Detalle por nivel"))

# ── Nubes de palabras clave ─────────────────────────────────
st.subheader("🔵 Palabras clave")

n1, n2 = st.columns(2)
with n1:
    st.write("**Palabras positivas**")
    if freq_pos:
        st.pyplot(nube_circular(dict(freq_pos), "Greens"))
    else:
        st.info("No se encontraron palabras positivas.")
with n2:
    st.write("**Palabras negativas**")
    if freq_neg:
        st.pyplot(nube_circular(dict(freq_neg), "Reds"))
    else:
        st.info("No se encontraron palabras negativas.")