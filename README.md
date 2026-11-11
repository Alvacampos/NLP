# NLP: Procesamiento de Lenguaje Natural

Examen de práctica de Procesamiento de Lenguaje Natural (Licenciatura en Inteligencia Artificial, UBP): análisis de una noticia periodística con un pipeline de PLN. Incluye preprocesamiento (tokenización, lematización y stopwords), frecuencias, keywords por TF-IDF, n-gramas, reconocimiento de entidades (NER), representación BoW y TF-IDF, y visualizaciones.

**Corpus:** "La NASA confirmó el final de Swift, el telescopio que caerá a la Tierra después de casi 22 años en el espacio". Infobae, sección Ciencia, 27/10/2026, por Cecilia Castro. [Enlace a la nota](https://www.infobae.com/america/ciencia-america/2026/10/27/la-nasa-confirmo-el-final-de-swift-el-telescopio-que-caera-a-la-tierra-despues-de-casi-22-anos-en-el-espacio/)

## Estructura del repositorio

| Archivo | Contenido |
|---|---|
| `Procesamiento de Lenguaje Natural-P1.pdf` | Parte 1: selección del corpus, categoría asignada (Tecnología) y análisis semántico con las hipótesis a validar. |
| `Procesamiento de Lenguaje Natural-P2.pdf` | Parte 2: informe completo con las figuras incluidas (corpus, código, análisis, visualizaciones y conclusiones). |
| `pln_swift.py` | Parte 2: script con todo el pipeline (corpus, preprocesamiento, técnicas de análisis y representación vectorial). |
| `informe_parte2.txt` | El mismo informe de la Parte 2 en texto plano, para pegar en la plataforma web. |
| `noticia.txt` | Texto de la noticia: título, bajada y un párrafo por línea (22 párrafos). |
| `requirements.txt` | Dependencias de Python. |
| `figuras/` | Visualizaciones generadas por el script: frecuencias, nube TF-IDF, heatmap TF-IDF, Ciencia vs. Tecnología y tabla de entidades. |

## Cómo ejecutar

Probado con Python 3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python pln_swift.py
```

`requirements.txt` incluye el modelo `es_core_news_md` de spaCy. La primera ejecución necesita conexión a internet: el script descarga la lista de stopwords de NLTK.

La salida se imprime por consola, organizada por secciones, y las figuras se guardan como PNG en `figuras/`.

## Pipeline

1. **Corpus:** la noticia se segmenta en 24 documentos (título, bajada y p01–p22). Con un único documento, IDF = log(1/1) = 0 y TF-IDF se anula.
2. **Preprocesamiento:** spaCy (`es_core_news_md`) analiza el texto original, se lematiza y después se normaliza cada lema (minúsculas, sin puntuación ni números). Se eliminan las stopwords de NLTK y los verbos auxiliares.
3. **Análisis:** top 20 lemas por frecuencia, keywords por TF-IDF (TF = f(t,d)/|d|, IDF = log(N/df)), bigramas y trigramas, NER sobre el texto original y métricas de estructura por segmento (densidad de keywords y proporción de vocabulario Ciencia/Tecnología).
4. **Representación:** BoW (`CountVectorizer`) y TF-IDF (`TfidfVectorizer`) con unigramas y bigramas, con su dispersión y la comparación de rankings.
5. **Visualizaciones:** cinco figuras PNG que resumen los resultados anteriores.

## Uso de herramientas de IA

La Parte 2 se desarrolló con la asistencia de Claude (Anthropic), un asistente de IA.

- **Autor:** eligió la noticia, asignó la categoría y realizó el análisis y las hipótesis de la Parte 1. También definió la consigna, el alcance y el método de trabajo, tomó las decisiones de preprocesamiento, y revisó y aprobó cada paso antes de avanzar al siguiente.
- **Asistente de IA:** escribió el código de `pln_swift.py`, generó las figuras y redactó el informe de la Parte 2 (`informe_parte2.txt` y su versión PDF), siempre bajo la supervisión del autor.
