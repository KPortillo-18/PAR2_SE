import streamlit as st
import pandas as pd
from textblob import TextBlob
from deep_translator import GoogleTranslator


# Configuración de la página 
st.set_page_config(
    page_title="Análisis de Sentimiento",
    page_icon="🧠",
    layout="wide",
)

# Comentarios internos por defecto (si no se sube archivo)
COMENTARIOS_INTERNOS = [
    "The car is amazing",
    "The car is poor",
    "The girl is pretty",
    "The service was terrible",
    "I love this product",
    "This is the worst experience ever",
    "The food was okay, nothing special",
    "Absolutely fantastic quality",
]


# Función principal 
def predict_sentiment(text, use_translation=False):
    if not text.strip():
        return None

    if use_translation:
        try:
            text = GoogleTranslator(source='es', target='en').translate(text)
        except Exception as e:
            st.warning(f"⚠️ Error al traducir, se analiza el texto original. ({e})")

    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity

    if polarity > 0.1:
        sentiment = "Positivo :D"
        emoji = "😊"
    elif polarity < -0.1:
        sentiment = "Negativo :("
        emoji = "😞"
    else:
        sentiment = "Neutral 😐"
        emoji = "😐"

    return {
        "sentiment": sentiment,
        "emoji": emoji,
        "polarity": polarity,
        "subjectivity": subjectivity,
    }


# Interfaz Streamlit 
st.title("🧠 Análisis de Sentimiento")
st.caption("Ingresa un texto y obtén su clasificación emocional al instante.")

with st.container(border=True):
    col_input, col_option = st.columns([3, 1])

    with col_input:
        user_text = st.text_area(
            "Texto a analizar",
            placeholder="Ej.: The car is amazing / Este producto es terrible…",
            height=120,
        )

    with col_option:
        use_translation = st.checkbox("🌐 Traducir ES→EN")

if user_text.strip():
    result = predict_sentiment(user_text, use_translation)
else:
    result = None

if result is None and not user_text.strip():
    st.info("✍️ Escribe un texto para comenzar el análisis.")
elif result is None:
    st.warning("⚠️ Por favor ingresa un texto válido.")
else:
    col_emoji, col_label = st.columns([1, 3])

    with col_emoji:
        st.markdown(f"### {result['emoji']}")

    with col_label:
        if "Positivo" in result["sentiment"]:
            st.success(result["sentiment"], icon="✅")
        elif "Negativo" in result["sentiment"]:
            st.error(result["sentiment"], icon="❌")
        else:
            st.info(result["sentiment"], icon="ℹ️")

    m1, m2 = st.columns(2)

    with m1:
        st.metric("Polaridad", f"{result['polarity']:.3f}", delta=f"rango [-1, 1]")
        bar_value = (result["polarity"] + 1) / 2
        st.progress(bar_value, text="← Negativo | Positivo →")

    with m2:
        st.metric("Subjetividad", f"{result['subjectivity']:.3f}", delta=f"rango [0, 1]")
        st.progress(result["subjectivity"], text="← Objetivo | Subjetivo →")

    with st.expander("📋 Detalle técnico"):
        st.json(result, expanded=False)


# Análisis por lote (archivo o comentarios internos)
st.divider()
st.header("📂 Análisis de un conjunto de comentarios")

archivo = st.file_uploader(
    "Sube un archivo .txt (un comentario por línea) o .csv (columna 'comentario')",
    type=["txt", "csv"],
)

use_translation_lote = st.checkbox("🌐 Traducir comentarios ES→EN", key="lote_trad")

if archivo is not None:
    if archivo.name.endswith(".csv"):
        df_archivo = pd.read_csv(archivo)
        columna = df_archivo.columns[0]
        comentarios = df_archivo[columna].astype(str).tolist()
    else:
        contenido = archivo.read().decode("utf-8")
        comentarios = [linea.strip() for linea in contenido.splitlines() if linea.strip()]
    st.success(f"✅ Se cargaron {len(comentarios)} comentarios del archivo.")
else:
    comentarios = COMENTARIOS_INTERNOS
    st.info("ℹ️ No se subió ningún archivo, se están usando comentarios internos de ejemplo.")

resultados = []
for comentario in comentarios:
    r = predict_sentiment(comentario, use_translation_lote)
    if r:
        resultados.append({
            "Comentario": comentario,
            "Sentimiento": r["sentiment"],
            "Polaridad": round(r["polarity"], 3),
            "Subjetividad": round(r["subjectivity"], 3),
        })

df_resultados = pd.DataFrame(resultados)

st.subheader("Resultados")
st.dataframe(df_resultados, use_container_width=True)

st.subheader("📊 Distribución de sentimientos")

conteo = df_resultados["Sentimiento"].value_counts()

col_g1, col_g2 = st.columns(2)

with col_g1:
    st.bar_chart(conteo)

with col_g2:
    st.write("Conteo de comentarios por sentimiento:")
    st.table(conteo)