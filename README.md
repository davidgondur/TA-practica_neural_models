# Práctica de Modelos Neuronales

Práctica de redes neuronales sobre transferencia de formalidad: "traducción" de textos vulgares/informales a formales.

## Dataset

[portex/multilingual-formality-transfer](https://huggingface.co/datasets/portex/multilingual-formality-transfer) (Hugging Face).

El dataset incluye varios idiomas, pero en esta práctica solo se usa el **portugués**.

## Entorno

- Gestor de paquetes: [uv](https://docs.astral.sh/uv/)
- Cuadernos: [marimo](https://marimo.io/)

```bash
uv sync                          # instalar dependencias
uv run marimo edit notebooks/    # abrir los cuadernos
```

## Estructura

```
notebooks/    # cuadernos marimo de la práctica
src/          # código del paquete
references/   # material de consulta (no versionado)
```
