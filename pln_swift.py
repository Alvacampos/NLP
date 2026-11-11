"""
Práctica PLN – Parte 2: Aplicación de PLN sobre el corpus elegido
Noticia: "La NASA confirmó el final de Swift..." (Infobae, 27/10/2026)

Secciones:
    1. Corpus            2. Preprocesamiento
    3. Técnicas de análisis (frecuencias, TF-IDF, n-gramas, NER, estructura)
    4. Representación vectorial (BoW y TF-IDF de scikit-learn)
    5. Visualizaciones (PNG en figuras/)
    6. Evidencia de limitaciones (números, negación, sinonimia, polisemia)

Requisitos:
    pip install -r requirements.txt
    python -m spacy download es_core_news_md

Entrada: noticia.txt (mismo directorio)
    Línea 1: título | Línea 2: bajada | Resto: un párrafo por línea
    (Se excluyen epígrafes de fotos e intertítulos: no son cuerpo de la nota.)
"""

import math
import re
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # genera PNG sin abrir ventanas
import matplotlib.pyplot as plt
import nltk
import pandas as pd
import spacy
from matplotlib.colors import LinearSegmentedColormap
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from wordcloud import WordCloud

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
RUTA_NOTICIA = Path(__file__).parent / "noticia.txt"
MODELO_SPACY = "es_core_news_md"

nltk.download("stopwords", quiet=True)
nlp = spacy.load(MODELO_SPACY)

# Stopwords: lista estándar de NLTK + lemas de verbos auxiliares/modales,
# que no aportan contenido temático (se aplican después de lematizar).
STOPWORDS = set(stopwords.words("spanish")) | {
    "ser", "estar", "haber", "tener", "hacer", "poder", "deber", "ir",
}

# Caracteres permitidos tras la limpieza: letras (con tildes, ñ, ü) y espacios.
# Elimina puntuación, símbolos y dígitos.
PATRON_NO_LETRAS = re.compile(r"[^a-záéíóúüñ\s]")


# ---------------------------------------------------------------------------
# 1. Corpus
# ---------------------------------------------------------------------------
def cargar_corpus(ruta: Path) -> list[dict]:
    """Lee la noticia y la segmenta: cada párrafo es un 'documento'.

    Con un solo documento el IDF es constante (log(1/1) = 0), por lo que
    se trata cada párrafo como documento para que TF-IDF sea informativo.
    """
    lineas = [l.strip() for l in ruta.read_text(encoding="utf-8").splitlines()]
    lineas = [l for l in lineas if l]
    corpus = []
    for i, texto in enumerate(lineas):
        seccion = "titulo" if i == 0 else "bajada" if i == 1 else f"p{i - 1:02d}"
        corpus.append({"id": seccion, "texto": texto})
    return corpus


# ---------------------------------------------------------------------------
# 2. Preprocesamiento
# ---------------------------------------------------------------------------
def normalizar(texto: str) -> str:
    """Minúsculas + eliminación de puntuación, símbolos y números.

    Se conservan las tildes: distinguen palabras ('él'/'el') y el
    lematizador de spaCy las necesita para funcionar correctamente.
    """
    texto = texto.lower()
    texto = PATRON_NO_LETRAS.sub(" ", texto)
    return re.sub(r"\s+", " ", texto).strip()


def preprocesar(texto: str) -> list[str]:
    """Pipeline: tokenización -> lematización -> normalización -> stopwords.

    spaCy analiza el texto ORIGINAL: las mayúsculas y la puntuación le dan
    contexto al etiquetador gramatical y mejoran la lematización (probado:
    sobre texto ya normalizado, 'Prueba' se lematizaba como el verbo 'probar').
    La normalización se aplica después, sobre cada lema.
    """
    doc = nlp(texto)                                   # tokenización
    tokens = []
    for t in doc:
        if t.is_punct or t.is_space or t.like_num:     # puntuación y números
            continue
        lema = normalizar(t.lemma_)                    # lematización + minúsculas
        # Un lema puede traer espacios: spaCy separa los clíticos
        # ('acercarse' -> 'acercar él') y la normalización convierte el guion
        # en espacio ('Domagal-Goldman'). Se divide en partes para que cada
        # token sea una sola palabra; 'él' cae luego como stopword.
        for parte in lema.split():
            if parte not in STOPWORDS and len(parte) > 1:     # stopwords
                tokens.append(parte)
    return tokens


def comparar_stemming_lematizacion(palabras: list[str]) -> list[tuple]:
    """Evidencia para justificar lematización frente a stemming."""
    stemmer = SnowballStemmer("spanish")
    return [(p, stemmer.stem(p), nlp(p)[0].lemma_) for p in palabras]


# ---------------------------------------------------------------------------
# 3. Técnicas de análisis
# ---------------------------------------------------------------------------
# Hipótesis de la Parte 1, expresadas en la forma de lema que produce el
# pipeline. Se escriben a mano porque lematizar palabras AISLADAS (sin
# contexto sintáctico) falla: 'archivo' -> 'archivar', 'órbita' -> 'órbitar'.
KEYWORDS_ESPERADAS = {
    "telescopio": ["telescopio"], "observatorio": ["observatorio"],
    "NASA": ["nasa"], "Swift": ["swift"], "órbita": ["órbita"],
    "altitud": ["altitud"], "reingreso": ["reingreso"],
    "atmósfera": ["atmósfera"],
    "resistencia atmosférica": ["resistencia", "atmosférico"],
    "satélite": ["satélite"], "nave robótica": ["nave", "robótico"],
    "propulsores": ["propulsor"], "actividad solar": ["actividad", "solar"],
    "clima espacial": ["clima", "espacial"], "rayos gamma": ["rayo", "gamma"],
    "supernovas": ["supernovas"],
    "estrellas de neutrones": ["estrella", "neutrón"],
    "misión espacial": ["misión", "espacial"], "archivo": ["archivo"],
    "terabytes": ["terabytes"],
}
BIGRAMAS_ESPERADOS = {
    "resistencia atmosférica": ("resistencia", "atmosférico"),
    "rayos gamma": ("rayo", "gamma"),
    "nave robótica": ("nave", "robótico"),
    "clima espacial": ("clima", "espacial"),
    "misión espacial": ("misión", "espacial"),
}
ENTIDADES_ESPERADAS = [
    "NASA", "Katalyst Space Technologies", "Centro de Vuelo Espacial Goddard",
    "Ghonhee Lee", "S. Bradley Cenko", "Shawn Domagal-Goldman", "Washington",
    "Observatorio Swift Neil Gehrels", "LINK", "GRB 050509B",
]

# Léxicos de dominio (SUPUESTO: definidos a priori por el analista, en forma
# de lema tal como los produce el pipeline). Se excluyen términos ambiguos
# entre ambos dominios (atmósfera, sol, solar, espacial, misión).
# 'telescopio'/'observatorio' se asignan a Tecnología: son el instrumento,
# coherente con la categoría asignada en la Parte 1.
LEXICO_CIENCIA = {
    "estallido", "rayo", "gamma", "supernovas", "estrella", "neutrón",
    "galaxia", "asteroides", "cometas", "agujero", "negro", "cosmo",
    "cósmico", "universo", "astronómico", "astrofísica", "científico",
    "hipótesis", "investigador", "descubrimiento", "espectro", "onda",
    "longitud", "fenómeno", "gravedad", "explosión", "resplandor",
    "hallazgo", "energético", "fusión", "artículo",
}
LEXICO_TECNOLOGIA = {
    "telescopio", "observatorio", "satélite", "nave", "órbita", "altitud",
    "altura", "propulsor", "robótico", "brazo", "mecánico", "reentrada",
    "reingresar", "reingresacer", "resistencia", "atmosférico",
    "instrumento", "dato", "archivo", "terabytes", "imagen", "ultravioleta",
    "óptico", "lanzar", "operar", "operación", "falla", "técnico", "empresa",
    "descenso", "estabilidad", "precisión", "apuntar", "vuelo", "trayecto",
}


def frecuencias(corpus: list[dict], n: int = 20) -> list[tuple[str, int]]:
    """Frecuencia absoluta de lemas en toda la noticia (enfoque BoW)."""
    return Counter(t for d in corpus for t in d["tokens"]).most_common(n)


def calcular_tfidf(corpus: list[dict]) -> pd.DataFrame:
    """TF-IDF según la definición de la materia (matriz segmento × término).

    TF(t,d) = f(t,d) / |d|      IDF(t) = log(N / df(t))   (log natural)
    Un término presente en todos los segmentos tendría IDF = 0. Se implementa
    a mano (sin suavizado) para respetar la fórmula de la guía; en el Paso 2
    se compara con la variante suavizada de scikit-learn.
    """
    n_docs = len(corpus)
    df = Counter(t for d in corpus for t in set(d["tokens"]))
    filas = {}
    for d in corpus:
        conteo, largo = Counter(d["tokens"]), len(d["tokens"])
        filas[d["id"]] = {t: (f / largo) * math.log(n_docs / df[t])
                          for t, f in conteo.items()}
    return pd.DataFrame.from_dict(filas, orient="index").fillna(0.0)


def keywords_noticia(tfidf: pd.DataFrame, n: int = 20) -> pd.Series:
    """Keywords a nivel noticia: TF-IDF medio de cada término sobre los
    segmentos. Premia términos con peso alto en varios párrafos sin estar
    en todos (los que aparecen en un solo párrafo largo quedan diluidos)."""
    return tfidf.mean(axis=0).sort_values(ascending=False).head(n)


def keywords_por_segmento(tfidf: pd.DataFrame, k: int = 3) -> dict:
    """Top-k términos por TF-IDF dentro de cada segmento."""
    return {seg: fila[fila > 0].sort_values(ascending=False).head(k)
            for seg, fila in tfidf.iterrows()}


def ngramas(corpus: list[dict], n: int) -> Counter:
    """N-gramas de lemas DENTRO de cada segmento (no cruzan párrafos).

    Se calculan sobre tokens sin stopwords: 'estallidos de rayos gamma'
    -> ('estallido', 'rayo', 'gamma'). Efecto colateral: pueden unirse
    palabras que en el original estaban separadas por una stopword o un
    número eliminado (adyacencias artificiales).
    """
    return Counter(
        tuple(d["tokens"][i:i + n])
        for d in corpus for i in range(len(d["tokens"]) - n + 1)
    )


def contrastar_ngramas(conteo: Counter, esperados: dict) -> list[tuple]:
    """Busca cada n-grama esperado (en forma de lema) en el conteo real."""
    return [(frase, " ".join(clave), conteo.get(clave, 0))
            for frase, clave in esperados.items()]


def extraer_entidades(corpus: list[dict]) -> pd.DataFrame:
    """NER de spaCy sobre el texto ORIGINAL de cada segmento.

    El NER depende de las mayúsculas y la puntuación (señales de nombre
    propio), por eso no se aplica sobre el texto normalizado.
    Etiquetas de es_core_news_md: PER, ORG, LOC, MISC.
    """
    filas = [(d["id"], ent.text, ent.label_)
             for d in corpus for ent in nlp(d["texto"]).ents]
    return pd.DataFrame(filas, columns=["segmento", "entidad", "tipo"])


def contrastar_entidades(entidades: pd.DataFrame,
                         esperadas: list[str]) -> list[tuple]:
    """Para cada entidad esperada: detección exacta, parcial o ausente."""
    resultado = []
    for esp in esperadas:
        exactas = entidades[entidades["entidad"] == esp]
        if not exactas.empty:
            tipos = ", ".join(sorted(exactas["tipo"].unique()))
            resultado.append((esp, "exacta", tipos))
            continue
        parciales = entidades[[e in esp or esp in e
                               for e in entidades["entidad"]]]
        if not parciales.empty:
            detalle = "; ".join(sorted({f"'{r.entidad}' {r.tipo}"
                                        for r in parciales.itertuples()}))
            resultado.append((esp, "parcial", detalle))
        else:
            resultado.append((esp, "NO detectada", "-"))
    return resultado


def estructura(corpus: list[dict], keywords: set[str]) -> pd.DataFrame:
    """Métricas por segmento para analizar la estructura de la noticia.

    - densidad_kw: proporción de tokens del segmento que pertenecen al set
      de keywords esperadas de la Parte 1 (mide dónde se concentra la señal).
    - prop_ciencia: tokens Ciencia / (Ciencia + Tecnología) según léxicos.
    """
    filas = []
    for d in corpus:
        toks = d["tokens"]
        n_cie = sum(t in LEXICO_CIENCIA for t in toks)
        n_tec = sum(t in LEXICO_TECNOLOGIA for t in toks)
        filas.append({
            "segmento": d["id"],
            "palabras": len(d["texto"].split()),
            "tokens": len(toks),
            "densidad_kw": sum(t in keywords for t in toks) / len(toks),
            "ciencia": n_cie,
            "tecnologia": n_tec,
            "prop_ciencia": n_cie / (n_cie + n_tec) if n_cie + n_tec else None,
        })
    return pd.DataFrame(filas).set_index("segmento")


# ---------------------------------------------------------------------------
# 4. Representación vectorial (BoW y TF-IDF con scikit-learn)
# ---------------------------------------------------------------------------
def identidad(x):
    """Función identidad: los documentos ya llegan tokenizados y lematizados.

    Se pasa como tokenizer y preprocessor para que scikit-learn NO vuelva a
    tokenizar ni a pasar a minúsculas (sus reglas por defecto descartarían
    el trabajo del pipeline de spaCy).
    """
    return x


def vectorizar(corpus: list[dict]) -> dict:
    """BoW (CountVectorizer) y TF-IDF (TfidfVectorizer), unigramas+bigramas.

    ngram_range=(1, 2): incorpora bigramas como 'rayo gamma' o 'resistencia
    atmosférico', que nombran conceptos que una palabra sola no captura.
    Los bigramas se forman dentro de cada segmento (cada uno es un documento).
    TfidfVectorizer usa por defecto una variante de la fórmula de la guía:
    TF = conteo bruto, IDF = ln((1+N)/(1+df)) + 1 (suavizado) y norma L2 por
    documento; por eso su ranking difiere del TF-IDF manual de la sección 3.
    """
    docs = [d["tokens"] for d in corpus]
    params = dict(tokenizer=identidad, preprocessor=identidad,
                  lowercase=False, token_pattern=None, ngram_range=(1, 2))
    resultado = {}
    for nombre, vec in [("BoW", CountVectorizer(**params)),
                        ("TF-IDF", TfidfVectorizer(**params))]:
        X = vec.fit_transform(docs)
        resultado[nombre] = pd.DataFrame(
            X.toarray(), index=[d["id"] for d in corpus],
            columns=vec.get_feature_names_out())
    return resultado


def describir_matriz(m: pd.DataFrame) -> dict:
    """Dimensiones y dispersión (% de celdas en cero) de una matriz doc×term."""
    bigramas = sum(" " in t for t in m.columns)
    return {"dimensiones": m.shape, "unigramas": m.shape[1] - bigramas,
            "bigramas": bigramas, "no_ceros": int((m.values != 0).sum()),
            "dispersion": float((m.values == 0).mean())}


def comparar_rankings(bow: pd.DataFrame, tfidf: pd.DataFrame,
                      n: int = 15) -> pd.DataFrame:
    """Ranking global por BoW (suma de conteos) vs TF-IDF (media por
    segmento). El rango 1 es el término más relevante."""
    puntaje = pd.DataFrame({"bow": bow.sum(axis=0),
                            "tfidf": tfidf.mean(axis=0)})
    puntaje["rank_bow"] = puntaje["bow"].rank(ascending=False, method="min")
    puntaje["rank_tfidf"] = puntaje["tfidf"].rank(ascending=False,
                                                  method="min")
    puntaje["delta"] = puntaje["rank_bow"] - puntaje["rank_tfidf"]
    return puntaje.sort_values("tfidf", ascending=False)


# ---------------------------------------------------------------------------
# 5. Visualizaciones
# ---------------------------------------------------------------------------
DIR_FIGURAS = Path(__file__).parent / "figuras"

# Paleta: dos tonos categóricos validados para daltonismo (azul/naranja),
# gris neutro para lo que no se destaca y una rampa azul secuencial
# (un solo tono, claro -> oscuro) para magnitudes.
AZUL, NARANJA, GRIS = "#2a78d6", "#eb6834", "#b8b7b0"
TINTA, TINTA_2 = "#0b0b0b", "#52514e"
RAMPA_AZUL = LinearSegmentedColormap.from_list(
    "azul", ["#fcfcfb", "#cde2fb", "#86b6ef", "#2a78d6", "#1c5cab", "#0d366b"])

# Evaluación MANUAL del NER (SUPUESTO: anotación del analista sobre la salida
# de spaCy). Clave (entidad, etiqueta). Lo no listado como correcto o con
# error de tipo/límites se considera falso positivo.
NER_CORRECTAS = {
    ("NASA", "ORG"), ("La NASA", "ORG"), ("Washington", "LOC"),
    ("Administración Nacional de Aeronáutica y el Espacio", "ORG"),
    ("la Tierra", "LOC"), ("Sol", "MISC"), ("LINK", "MISC"),
    ("GRB 050509B", "MISC"), ("Ghonhee Lee", "PER"), ("Cenko", "PER"),
    ("Shawn Domagal-Goldman", "PER"),
}
NER_TIPO_ERRONEO = {  # entidad real, etiqueta incorrecta
    ("Swift", "PER"): "objeto (telescopio)",
    ("Swift", "LOC"): "objeto (telescopio)",
    ("Observatorio Swift", "LOC"): "objeto (telescopio)",
    ("Neil Gehrels", "PER"): "parte del nombre de un objeto",
    ("Katalyst Space Technologies", "PER"): "ORG (empresa)",
    ("División de Astrofísica", "MISC"): "ORG (división de la NASA)",
}
NER_LIMITES_ERRONEOS = {  # etiqueta plausible, span cortado o ampliado
    ("Centro", "LOC"): "fragmento de 'Centro de Vuelo Espacial Goddard'",
    ("Vuelo Espacial Goddard", "MISC"): "fragmento del mismo nombre",
    ("Bradley Cenko", "PER"): "falta la inicial 'S.'",
    ("telescopio espacial Swift", "MISC"): "incluye sustantivos comunes",
}


def evaluar_ner(entidades: pd.DataFrame) -> pd.DataFrame:
    """Agrega las menciones por (entidad, tipo) y les asigna la evaluación
    manual: correcta, tipo erróneo, límites erróneos o falso positivo."""
    tabla = (entidades.groupby(["entidad", "tipo"]).size()
             .rename("frec").reset_index())

    def clasificar(fila):
        clave = (fila["entidad"], fila["tipo"])
        if clave in NER_CORRECTAS:
            return "correcta", ""
        if clave in NER_TIPO_ERRONEO:
            return "tipo erróneo", NER_TIPO_ERRONEO[clave]
        if clave in NER_LIMITES_ERRONEOS:
            return "límites erróneos", NER_LIMITES_ERRONEOS[clave]
        return "falso positivo", ""

    tabla[["evaluacion", "nota"]] = tabla.apply(
        clasificar, axis=1, result_type="expand")
    return tabla


def _estilo_ejes(ax) -> None:
    """Ejes y grilla recesivos: el dato es lo que tiene que destacar."""
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(GRIS)
    ax.tick_params(colors=TINTA_2, labelsize=9)
    ax.set_axisbelow(True)


def fig_frecuencias(corpus: list[dict], kw_esperadas: set[str],
                    ruta: Path) -> None:
    """Barras horizontales del top 20 de lemas; en azul las keywords que la
    Parte 1 había anticipado (validación visual de la hipótesis)."""
    datos = frecuencias(corpus)[::-1]
    terminos, valores = zip(*datos)
    colores = [AZUL if t in kw_esperadas else GRIS for t in terminos]
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(terminos, valores, color=colores, height=0.7,
            edgecolor="white", linewidth=1)
    for y, v in enumerate(valores):
        ax.text(v + 0.15, y, str(v), va="center", fontsize=8, color=TINTA_2)
    _estilo_ejes(ax)
    ax.xaxis.grid(True, color="#e6e5e0", linewidth=0.8)
    ax.set_xlabel("Frecuencia absoluta en la noticia", color=TINTA_2)
    ax.legend(handles=[
        plt.Rectangle((0, 0), 1, 1, color=AZUL,
                      label="Keyword esperada (Parte 1)"),
        plt.Rectangle((0, 0), 1, 1, color=GRIS, label="Otro lema")],
        loc="lower right", frameon=False, fontsize=9)
    ax.set_title("Top 20 lemas por frecuencia (BoW)", loc="left",
                 fontsize=13, color=TINTA, pad=12)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def fig_nube(tfidf_sklearn: pd.DataFrame, ruta: Path) -> None:
    """Nube de palabras ponderada por TF-IDF medio (unigramas y bigramas).
    Se usa TF-IDF y no la frecuencia para que el tamaño refleje relevancia
    y no solo repetición."""
    pesos = tfidf_sklearn.mean(axis=0)
    pesos = pesos[pesos > 0].sort_values(ascending=False).head(80)

    def color(palabra, **_):
        # Tono por peso: más oscuro = más relevante (rampa secuencial).
        rel = pesos[palabra] / pesos.max()
        return matplotlib.colors.to_hex(RAMPA_AZUL(0.45 + 0.55 * rel))

    nube = WordCloud(width=1600, height=900, background_color="white",
                     prefer_horizontal=1.0, color_func=color,
                     random_state=42, margin=6).generate_from_frequencies(
                         pesos.to_dict())
    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.imshow(nube, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("Nube de términos ponderada por TF-IDF (unigramas y "
                 "bigramas)", loc="left", fontsize=13, color=TINTA, pad=10)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def fig_heatmap(tfidf_sklearn: pd.DataFrame, ruta: Path, n: int = 20) -> None:
    """Heatmap segmento × término (top n por TF-IDF medio). Muestra en qué
    párrafos se concentra cada término: la distribución de la señal."""
    top = tfidf_sklearn.mean(axis=0).sort_values(ascending=False).head(n)
    m = tfidf_sklearn[top.index]
    fig, ax = plt.subplots(figsize=(11, 8.5))
    im = ax.imshow(m.values, cmap=RAMPA_AZUL, aspect="auto",
                   vmin=0, vmax=m.values.max())
    ax.set_xticks(range(n), m.columns, rotation=55, ha="right", fontsize=9,
                  color=TINTA_2)
    ax.set_yticks(range(len(m)), m.index, fontsize=9, color=TINTA_2)
    # Separadores blancos entre celdas (2 px) para leer filas y columnas.
    ax.set_xticks([x - 0.5 for x in range(1, n)], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(m))], minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    barra = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    barra.set_label("Peso TF-IDF (scikit-learn, norma L2)", color=TINTA_2)
    barra.outline.set_visible(False)
    ax.set_title(f"TF-IDF por segmento: top {n} términos", loc="left",
                 fontsize=13, color=TINTA, pad=12)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def fig_ciencia_tecnologia(est: pd.DataFrame, ruta: Path) -> None:
    """Barras apiladas por segmento con los tokens de cada léxico de dominio.
    Hace visible el giro temático de la nota hacia 'Ciencia'."""
    x = range(len(est))
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x, est["tecnologia"], color=AZUL, width=0.75,
           edgecolor="white", linewidth=1, label="Léxico Tecnología")
    ax.bar(x, est["ciencia"], bottom=est["tecnologia"], color=NARANJA,
           width=0.75, edgecolor="white", linewidth=1,
           label="Léxico Ciencia")
    corte = list(est.index).index("p13") - 0.5
    ax.axvline(corte, color=TINTA_2, linewidth=1, linestyle="--")
    alto = (est["ciencia"] + est["tecnologia"]).max()
    ax.text(corte - 0.3, alto + 0.6, "operación y caída  ←", ha="right",
            fontsize=9, color=TINTA_2)
    ax.text(corte + 0.3, alto + 0.6, "→  logros astrofísicos", ha="left",
            fontsize=9, color=TINTA_2)
    ax.set_xticks(list(x), est.index, rotation=45, ha="right")
    _estilo_ejes(ax)
    ax.yaxis.grid(True, color="#e6e5e0", linewidth=0.8)
    ax.set_ylim(0, alto + 2)
    ax.set_ylabel("Tokens del léxico de dominio", color=TINTA_2)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.set_title("Vocabulario Ciencia vs. Tecnología por segmento",
                 loc="left", fontsize=13, color=TINTA, pad=12)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def fig_tabla_entidades(evaluacion: pd.DataFrame, ruta: Path) -> None:
    """Tabla de entidades de spaCy con su evaluación manual. Los falsos
    positivos se resumen en una fila para mantener la tabla legible."""
    orden = {"correcta": 0, "tipo erróneo": 1, "límites erróneos": 2}
    visibles = evaluacion[evaluacion["evaluacion"] != "falso positivo"].copy()
    visibles["o"] = visibles["evaluacion"].map(orden)
    visibles = visibles.sort_values(["o", "tipo", "frec"],
                                    ascending=[True, True, False])
    fp = evaluacion[evaluacion["evaluacion"] == "falso positivo"]
    filas = [[r.entidad, r.tipo, str(r.frec), r.evaluacion, r.nota]
             for r in visibles.itertuples()]
    filas.append([f"{len(fp)} frases al inicio de oración "
                  "('Esa fuerza', 'Según', ...)",
                  "MISC", str(fp["frec"].sum()), "falso positivo",
                  "mayúscula inicial confundida con nombre propio"])
    fig, ax = plt.subplots(figsize=(12, 0.27 * len(filas) + 0.6))
    ax.axis("off")
    tabla = ax.table(cellText=filas, loc="upper center", cellLoc="left",
                     colLabels=["Entidad (texto original)", "Etiqueta",
                                "Frec.", "Evaluación", "Observación"],
                     colWidths=[0.36, 0.08, 0.06, 0.13, 0.37])
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(8.5)
    tabla.scale(1, 1.35)
    fondo = {"correcta": "#ffffff", "tipo erróneo": "#fbe3d9",
             "límites erróneos": "#fdf0d0", "falso positivo": "#ececea"}
    for (fila, _), celda in tabla.get_celld().items():
        celda.set_edgecolor("#e6e5e0")
        if fila == 0:
            celda.set_text_props(weight="bold", color=TINTA)
            celda.set_facecolor("#f3f2ee")
        else:
            celda.set_facecolor(fondo[filas[fila - 1][3]])
            celda.set_text_props(color=TINTA)
    total = evaluacion["frec"].sum()
    ok = evaluacion.loc[evaluacion["evaluacion"] == "correcta", "frec"].sum()
    ax.set_title(f"Entidades detectadas por spaCy (es_core_news_md): "
                 f"{ok} de {total} menciones correctas", loc="left",
                 fontsize=13, color=TINTA)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)


def generar_figuras(corpus: list[dict]) -> list[Path]:
    """Paso 3: genera las figuras en figuras/ y devuelve sus rutas."""
    DIR_FIGURAS.mkdir(exist_ok=True)
    kw = {t for lemas in KEYWORDS_ESPERADAS.values() for t in lemas}
    tfidf = vectorizar(corpus)["TF-IDF"]
    rutas = {
        "01_frecuencias_top20.png":
            lambda r: fig_frecuencias(corpus, kw, r),
        "02_nube_tfidf.png": lambda r: fig_nube(tfidf, r),
        "03_heatmap_tfidf.png": lambda r: fig_heatmap(tfidf, r),
        "04_ciencia_vs_tecnologia.png":
            lambda r: fig_ciencia_tecnologia(estructura(corpus, kw), r),
        "05_entidades_ner.png":
            lambda r: fig_tabla_entidades(
                evaluar_ner(extraer_entidades(corpus)), r),
    }
    for nombre, generar in rutas.items():
        generar(DIR_FIGURAS / nombre)
    return [DIR_FIGURAS / n for n in rutas]


# ---------------------------------------------------------------------------
# 6. Evidencia de limitaciones de la representación
# ---------------------------------------------------------------------------
NEGACIONES = {"no", "ni", "sin", "tampoco", "nunca", "jamás"}
PARES_SINONIMOS = [  # misma referencia o significado cercano en la nota
    ("telescopio", "observatorio"), ("telescopio", "satélite"),
    ("altura", "altitud"), ("reentrada", "reingresar"),
    ("estallido", "explosión"),
]


def numeros_eliminados(corpus: list[dict]) -> list[tuple[str, str]]:
    """Magnitudes que el preprocesamiento descarta (like_num de spaCy)."""
    return [(d["id"], t.text) for d in corpus for t in nlp(d["texto"])
            if t.like_num]


def negaciones_perdidas(corpus: list[dict]) -> list[tuple[str, str, bool]]:
    """Negaciones del original y si sobreviven al filtro de stopwords.
    Se devuelve también el núcleo sintáctico al que modifican."""
    return [(d["id"], f"{t.text} -> {t.head.text}", t.lower_ in STOPWORDS)
            for d in corpus for t in nlp(d["texto"])
            if t.lower_ in NEGACIONES]


def similitud_sinonimos(bow: pd.DataFrame) -> list[tuple]:
    """Compara, para pares de sinónimos, tres medidas:
    - one-hot: siempre 0 (vectores ortogonales: cada término es una dimensión)
    - coseno entre columnas BoW: solo refleja co-ocurrencia en segmentos
    - coseno entre embeddings de spaCy (300 dimensiones, estáticos)
    """
    resultado = []
    for a, b in PARES_SINONIMOS:
        va, vb = bow[a].values, bow[b].values
        cos_bow = float(va @ vb / ((va @ va) ** 0.5 * (vb @ vb) ** 0.5))
        resultado.append((a, b, cos_bow, nlp.vocab[a].similarity(nlp.vocab[b])))
    return resultado


# ---------------------------------------------------------------------------
# Reportes por consola
# ---------------------------------------------------------------------------
def titulo(texto: str) -> None:
    print(f"\n{'=' * 70}\n{texto}\n{'=' * 70}")


def reportar_analisis(corpus: list[dict]) -> None:
    """Paso 1: imprime los resultados de las cinco técnicas de análisis."""
    kw_esperadas = {t for lemas in KEYWORDS_ESPERADAS.values() for t in lemas}
    frec_total = Counter(t for d in corpus for t in d["tokens"])

    titulo("1a. FRECUENCIA DE TÉRMINOS (top 20 lemas)")
    for i, (t, f) in enumerate(frecuencias(corpus), 1):
        marca = "*" if t in kw_esperadas else " "
        print(f"{i:>2}. {marca} {t:<16}{f:>3}")
    print("   (* = lema presente en las keywords esperadas de la Parte 1)")

    titulo("1b. KEYWORDS POR TF-IDF")
    tfidf = calcular_tfidf(corpus)
    print("Nivel noticia (TF-IDF medio sobre 24 segmentos):")
    for i, (t, v) in enumerate(keywords_noticia(tfidf).items(), 1):
        marca = "*" if t in kw_esperadas else " "
        print(f"{i:>2}. {marca} {t:<16}{v:.4f}")
    print("\nPor segmento (top 3):")
    for seg, serie in keywords_por_segmento(tfidf).items():
        print(f"  {seg:<7}" + ", ".join(f"{t} ({v:.3f})"
                                        for t, v in serie.items()))
    ranking = keywords_noticia(tfidf, n=len(tfidf.columns))
    posicion = {t: i for i, t in enumerate(ranking.index, 1)}
    print(f"\nContraste keywords esperadas (Parte 1) — vocabulario: "
          f"{len(tfidf.columns)} lemas:")
    for frase, lemas in KEYWORDS_ESPERADAS.items():
        detalle = ", ".join(
            f"{t} f={frec_total[t]} rank={posicion.get(t, '-')}"
            for t in lemas)
        print(f"  {frase:<26}{detalle}")

    titulo("1c. N-GRAMAS")
    bi, tri = ngramas(corpus, 2), ngramas(corpus, 3)
    print(f"Bigramas distintos: {len(bi)} (con frec >= 2: "
          f"{sum(v >= 2 for v in bi.values())})")
    for g, f in bi.most_common(10):
        print(f"  {' '.join(g):<32}{f}")
    print(f"\nTrigramas distintos: {len(tri)} (con frec >= 2: "
          f"{sum(v >= 2 for v in tri.values())})")
    for g, f in tri.most_common(10):
        print(f"  {' '.join(g):<40}{f}")
    print("\nContraste con bigramas esperados:")
    for frase, lema, f in contrastar_ngramas(bi, BIGRAMAS_ESPERADOS):
        print(f"  {frase:<26}-> {lema:<26}frec = {f}")

    titulo("1d. NER (spaCy sobre texto original)")
    ents = extraer_entidades(corpus)
    print(f"Menciones: {len(ents)} | entidades únicas: "
          f"{ents['entidad'].nunique()}")
    print(ents.groupby("tipo").size().rename("menciones").to_string())
    tabla = (ents.groupby(["tipo", "entidad"]).size().rename("frec")
             .reset_index().sort_values(["tipo", "frec"],
                                        ascending=[True, False]))
    print("\n" + tabla.to_string(index=False))
    print("\nContraste con entidades esperadas:")
    for esp, estado, detalle in contrastar_entidades(ents,
                                                     ENTIDADES_ESPERADAS):
        print(f"  {esp:<34}{estado:<14}{detalle}")

    titulo("1e. ESTRUCTURA COMPUTACIONAL")
    est = estructura(corpus, kw_esperadas)
    print(est.to_string(float_format=lambda x: f"{x:.2f}"))
    cab, cuerpo = est.loc[["titulo", "bajada"]], est.drop(["titulo", "bajada"])
    print(f"\nDensidad de keywords: título+bajada = "
          f"{cab['densidad_kw'].mean():.2f} | párrafos (media) = "
          f"{cuerpo['densidad_kw'].mean():.2f}")
    tot_c, tot_t = est["ciencia"].sum(), est["tecnologia"].sum()
    print(f"Vocabulario de dominio total: Ciencia = {tot_c}, "
          f"Tecnología = {tot_t} ({tot_c / (tot_c + tot_t):.0%} Ciencia)")
    # Corte en p13: allí la nota pasa de la operación/caída del telescopio
    # (p01-p12) a la historia y los logros científicos de la misión (p13-p22).
    corte = list(cuerpo.index).index("p13")
    for nombre, parte in [("p01-p12", cuerpo.iloc[:corte]),
                          ("p13-p22", cuerpo.iloc[corte:])]:
        c, t = parte["ciencia"].sum(), parte["tecnologia"].sum()
        print(f"  {nombre}: Ciencia = {c}, Tecnología = {t} "
              f"({c / (c + t):.0%} Ciencia)")


def reportar_representacion(corpus: list[dict]) -> None:
    """Paso 2: BoW vs TF-IDF de scikit-learn."""
    titulo("2. REPRESENTACIÓN: BoW vs TF-IDF (scikit-learn, ngram_range=(1,2))")
    mats = vectorizar(corpus)
    for nombre, m in mats.items():
        info = describir_matriz(m)
        print(f"{nombre:<7} dimensiones {info['dimensiones'][0]} × "
              f"{info['dimensiones'][1]} ({info['unigramas']} unigramas + "
              f"{info['bigramas']} bigramas) | no ceros = {info['no_ceros']}"
              f" | dispersión = {info['dispersion']:.1%}")

    bow, tfidf = mats["BoW"], mats["TF-IDF"]
    rk = comparar_rankings(bow, tfidf)
    rho = rk["bow"].corr(rk["tfidf"], method="spearman")
    print(f"\nCorrelación de Spearman entre rankings (todos los términos): "
          f"{rho:.2f}")

    top_bow = rk.sort_values("bow", ascending=False).head(15)
    top_tfidf = rk.head(15)
    print(f"\n{'#':>2}  {'BoW (suma conteos)':<30}{'TF-IDF (media)':<34}")
    for i, ((tb, fb), (tt, ft)) in enumerate(
            zip(top_bow.iterrows(), top_tfidf.iterrows()), 1):
        print(f"{i:>2}  {tb:<22}{int(fb['bow']):>3}     "
              f"{tt:<24}{ft['tfidf']:.4f}  (BoW #{int(ft['rank_bow'])})")
    comunes = set(top_bow.index) & set(top_tfidf.index)
    print(f"\nTérminos compartidos en ambos top 15: {len(comunes)}")
    print("Solo en top BoW:   ", sorted(set(top_bow.index) - comunes))
    print("Solo en top TF-IDF:", sorted(set(top_tfidf.index) - comunes))

    bigr = rk[[" " in t for t in rk.index]]
    print("\nTop 5 bigramas por TF-IDF:",
          ", ".join(f"{t} (#{int(r)})" for t, r in
                    bigr.head(5)["rank_tfidf"].items()))

    manual = keywords_noticia(calcular_tfidf(corpus)).index
    solo_uni = rk[[" " not in t for t in rk.index]].head(20).index
    print(f"\nTop 20 unigramas: TF-IDF manual (guía) vs scikit-learn — "
          f"coincidencias: {len(set(manual) & set(solo_uni))}/20")


def reportar_limitaciones(corpus: list[dict]) -> None:
    """Paso 4: cuantifica lo que la representación BoW/TF-IDF deja afuera."""
    titulo("4. EVIDENCIA DE LIMITACIONES")
    nums = numeros_eliminados(corpus)
    print(f"Magnitudes numéricas eliminadas: {len(nums)} -> "
          + ", ".join(f"{t} ({s})" for s, t in nums))

    negs = negaciones_perdidas(corpus)
    perdidas = sum(es_stop for *_, es_stop in negs)
    print(f"\nNegaciones en el original: {len(negs)}; eliminadas como "
          f"stopword: {perdidas}")
    for seg, rel, es_stop in negs:
        print(f"  {seg:<5}{rel:<26}{'eliminada' if es_stop else 'conservada'}")

    print("\nSinonimia (coseno): one-hot | columnas BoW | embeddings spaCy")
    for a, b, c_bow, c_emb in similitud_sinonimos(vectorizar(corpus)["BoW"]):
        print(f"  {a + ' / ' + b:<28}0.00 | {c_bow:.2f} | {c_emb:.2f}")

    print("\nPolisemia de 'alto' (lema y palabra que modifica):")
    usos = [f"{t.text} {t.head.text} ({d['id']})" for d in corpus
            for t in nlp(d["texto"]) if t.lemma_.lower() == "alto"]
    print("  " + "; ".join(usos))


def reportar_corpus(corpus: list[dict]) -> None:
    """Paso 0: métricas del corpus y evidencia del preprocesamiento."""
    palabras_originales = sum(len(d["texto"].split()) for d in corpus)
    todos_tokens = [t for d in corpus for t in d["tokens"]]

    print("=== CORPUS ===")
    print(f"Segmentos (documentos): {len(corpus)}")
    print(f"Palabras originales:    {palabras_originales}")
    print(f"Tokens tras limpieza:   {len(todos_tokens)} "
          f"({1 - len(todos_tokens) / palabras_originales:.0%} de reducción)")
    print(f"Vocabulario único:      {len(set(todos_tokens))}")

    print("\n=== EJEMPLO DE PREPROCESAMIENTO (título) ===")
    print("Original:   ", corpus[0]["texto"])
    print("Normalizado:", normalizar(corpus[0]["texto"]))
    print("Tokens:     ", corpus[0]["tokens"])

    print("\n=== STEMMING vs LEMATIZACIÓN ===")
    muestra = ["telescopios", "orbitaba", "reingresará", "atmosférica",
               "propulsores", "estallidos", "científicos", "cayó"]
    print(f"{'palabra':<14}{'stem':<14}{'lema':<14}")
    for p, s, l in comparar_stemming_lematizacion(muestra):
        print(f"{p:<14}{s:<14}{l:<14}")


# ---------------------------------------------------------------------------
# Ejecución
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    corpus = cargar_corpus(RUTA_NOTICIA)
    for d in corpus:
        d["tokens"] = preprocesar(d["texto"])

    reportar_corpus(corpus)       # Paso 0
    reportar_analisis(corpus)     # Paso 1
    reportar_representacion(corpus)  # Paso 2

    titulo("3. VISUALIZACIONES")    # Paso 3
    for ruta in generar_figuras(corpus):
        print(f"Guardada: {ruta.relative_to(ruta.parent.parent)}")

    reportar_limitaciones(corpus)  # Paso 4 (evidencia para la interpretación)
