import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Análisis exploratorio de datos (EDA)

    Exploración del dataset [portex/multilingual-formality-transfer](https://huggingface.co/datasets/portex/multilingual-formality-transfer), que contiene pares de textos informales/vulgares y su versión formal en varios idiomas. En esta práctica solo se usará el **portugués**.
    """)
    return


@app.cell
def _():
    import altair as alt
    import polars as pl

    return (pl,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Carga del dataset
    """)
    return


@app.cell
def _(pl):
    # El dataset es un único parquet; polars lo lee directamente desde el Hub
    # (requiere estar autenticado, p. ej. con `uv run hf auth login`)
    DATASET_URL = "hf://datasets/portex/multilingual-formality-transfer/formal_informal_translations_sample.parquet"

    df_raw = pl.read_parquet(DATASET_URL)

    print(f"Filas: {df_raw.height:,}")
    print(f"Columnas: {df_raw.columns}")
    df_raw.head()
    return (df_raw,)


@app.cell
def _(df_raw):
    df_raw.schema
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Idiomas disponibles

    Comprobamos los idiomas disponibles en el dataset:
    """)
    return


@app.cell
def _(df_raw, pl):
    (df_raw.group_by("language")
        .agg(pl.len().alias("n"))
        .with_columns((pl.col("n") / pl.col("n").sum() * 100).alias("%"))
        .sort("n", descending=True))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Hay 9 idiomas en el dataset. El portugués es el que tiene la mayor representación, con 304.531 instancias (un 77.5 % de las instancias).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Filtrado: portugués
    """)
    return


@app.cell
def _(df_raw, pl):
    df_pt = df_raw.filter(pl.col("language")=="Portuguese")
    df_pt
    return (df_pt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Calidad de los datos
    """)
    return


@app.cell
def _(df_pt, pl):
    # Columnas de texto: todas las de tipo string salvo la de idioma
    TEXT_COLS = [c for c, t in df_pt.schema.items() if c != "language" and t == pl.String]

    _n_dup = df_pt.height - df_pt.unique(subset=TEXT_COLS).height
    print(f"Columnas de texto: {TEXT_COLS}")
    print(f"Filas duplicadas: {_n_dup:,}")
    print(f"Valores nulos por columna: {df_pt.null_count()}")
    return (TEXT_COLS,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Longitud de los textos
    """)
    return


@app.cell
def _(TEXT_COLS, df_pt, pl):
    lengths = df_pt.select(
        *[pl.col(c).fill_null("").str.len_chars().alias(f"{c}_chars") for c in TEXT_COLS],
        *[pl.col(c).fill_null("").str.count_matches(r"\S+").alias(f"{c}_words") for c in TEXT_COLS],
    )
    lengths.describe()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Nubes de palabras
    """)
    return


@app.cell
def _(df_pt):
    from wordcloud import WordCloud

    # Stopwords básicas del portugués (las de wordcloud solo cubren el inglés)
    STOPWORDS_PT = set(
        """
        a à ao aos as às até com como da das de dela delas dele deles do dos e é
        ela elas ele eles em entre era essa essas esse esses esta está estão estas
        este estes eu foi for há isso isto já la lhe lhes mais mas me mesmo meu
        minha muito na não nas nem no nos nós nossa nosso num numa o os ou para
        pela pelas pelo pelos por qual quando que quem se sem ser seu seus sua
        suas são só também te tem têm ter teu tu tua um uma umas uns vai você
        vocês vos tá pra pro q
        """.split()
    )

    def nube(col):
        texto = " ".join(df_pt.get_column(col).drop_nulls().to_list()).lower()
        return WordCloud(
            width=800,
            height=400,
            background_color="white",
            stopwords=STOPWORDS_PT,
            collocations=False,
            random_state=0,
        ).generate(texto)

    return (nube,)


@app.cell
def _(mo, nube):
    mo.vstack(
        [
            mo.md("### `original_text`"),
            mo.image(nube("original_text").to_image()),
            mo.md("### `formal_text`"),
            mo.image(nube("formal_text").to_image()),
        ]
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
