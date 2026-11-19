# /// script
# requires-python = ">=3.12,<3.15"
# dependencies = [
#     "accelerate",
#     "bitsandbytes",
#     "datasets",
#     "evaluate",
#     "marimo>=0.25.2",
#     "numpy>=2",
#     "peft",
#     "sacrebleu",
#     "torch==2.15.0",
#     "transformers",
#     "unbabel-comet==2.2.7",
#     "setuptools<81",
# ]
#
# [tool.uv]
# override-dependencies = ["numpy>=2"]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np

    return mo, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Baseline: NLLB para transferencia de formalidad en portugués
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    En este cuaderno usamos el modelo preentrenado de Meta [NLLB](https://huggingface.co/docs/transformers/model_doc/nllb), sin ningún ajuste, sobre el dataset [portex/multilingual-formality-transfer](https://huggingface.co/datasets/portex/multilingual-formality-transfer). El dataset contiene pares de textos informales/coloquiales (`original_text`) y su versión formal (`formal_text`) en el mismo idioma. Solo usaremos el **portugués**.

    La tarea se plantea como una "traducción" monolingüe portugués → portugués (informal → formal). NLLB no se entrenó para esto, así que el baseline mide cuánto se acerca el modelo a la referencia formal sin ningún ajuste.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Carga del dataset

    El dataset es privado/con acceso restringido en el Hub, por lo que hace falta un token (`HF_TOKEN` en `.env`). Se distribuye como un único fichero parquet con una sola partición (`train`).
    """)
    return


@app.cell
def _():
    from datasets import load_dataset

    raw_dataset = load_dataset("portex/multilingual-formality-transfer", split="train")

    print(raw_dataset)
    return (raw_dataset,)


@app.cell
def _(raw_dataset):
    raw_dataset.features
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Cada ejemplo tiene un identificador (`custom_id`), el idioma (`language`), el texto original informal (`original_text`) y su versión formal (`formal_text`), generada con GPT-4o-mini.

    Filtramos solo el portugués:
    """)
    return


@app.cell
def _(raw_dataset):
    raw_dataset_pt = raw_dataset.filter(lambda x: x["language"] == "Portuguese")
    print(raw_dataset_pt)
    return (raw_dataset_pt,)


@app.cell
def _(raw_dataset_pt):
    raw_dataset_pt[:3]["original_text"]
    return


@app.cell
def _(raw_dataset_pt):
    raw_dataset_pt[:3]["formal_text"]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Particiones

    A diferencia de Europarl-ST, este dataset no trae particiones predefinidas. Creamos entrenamiento, validación y test con una semilla fija para que los demás cuadernos (fine-tuning, LLAMA...) usen exactamente las mismas particiones. Por limitaciones de cómputo, validación y test se fijan en un tamaño pequeño.
    """)
    return


@app.cell
def _(raw_dataset_pt):
    from datasets import DatasetDict

    seed = 42
    n_valid = 45_000
    n_test = 60_000

    _split = raw_dataset_pt.train_test_split(test_size=n_valid + n_test, seed=seed)
    _valid_test = _split["test"].train_test_split(test_size=n_test, seed=seed)

    raw_datasets = DatasetDict(
        {
            "train": _split["train"],
            "validation": _valid_test["train"],
            "test": _valid_test["test"],
        }
    )
    print(raw_datasets)
    return (raw_datasets,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Tokenización

    Cargamos el tokenizador preentrenado de NLLB. El idioma de origen y de destino es el mismo: portugués (`por_Latn`).

    Los textos de este dataset son bastante más largos que los de Europarl-ST (149 caracteres de media en el original y 183 en el formal), así que subimos la longitud máxima a 64 tokens.
    """)
    return


@app.cell
def _():
    max_tok_length = 64

    from transformers import AutoTokenizer

    checkpoint = "facebook/nllb-200-distilled-600M"
    src_code = "por_Latn"
    tgt_code = "por_Latn"
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint,
        padding=True,
        pad_to_multiple_of=8,
        src_lang=src_code,
        tgt_lang=tgt_code,
        truncation=True,
        max_length=max_tok_length,
    )
    return checkpoint, max_tok_length, tgt_code, tokenizer


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Cada par se preprocesa con el tokenizador: `original_text` es la entrada y `formal_text` la salida esperada (`labels`).
    """)
    return


@app.cell
def _(tokenizer):
    def preprocess_function(sample):
        model_inputs = tokenizer(
            sample["original_text"],
            text_target=sample["formal_text"],
        )
        return model_inputs

    return (preprocess_function,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Comprobamos lo que hace `preprocess_function` con una muestra pequeña:
    """)
    return


@app.cell
def _(preprocess_function, raw_datasets):
    _sample = raw_datasets["train"].select(range(2))
    model_input = preprocess_function(
        {
            "original_text": list(_sample["original_text"]),
            "formal_text": list(_sample["formal_text"]),
        }
    )
    print(model_input)
    return (model_input,)


@app.cell
def _(model_input, tokenizer):
    for _sample in model_input["input_ids"]:
        print(tokenizer.convert_ids_to_tokens(_sample))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Podemos recuperar el texto con `batch_decode`:
    """)
    return


@app.cell
def _(model_input, tokenizer):
    tokenizer.batch_decode(model_input["input_ids"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Aplicamos `preprocess_function` a las tres particiones:
    """)
    return


@app.cell
def _(preprocess_function, raw_datasets):
    tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)
    return (tokenized_datasets,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Descartamos los pares en los que el texto original o el formal superan el número máximo de tokens:
    """)
    return


@app.cell
def _(max_tok_length, tokenized_datasets):
    tokenized_datasets_1 = tokenized_datasets.filter(
        lambda x: len(x["input_ids"]) <= max_tok_length
        and len(x["labels"]) <= max_tok_length,
        desc=f"Descartando textos con más de {max_tok_length} tokens",
    )
    print(tokenized_datasets_1)
    return (tokenized_datasets_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Histograma de longitudes (en tokens) del texto original tras el filtrado:
    """)
    return


@app.cell
def _(max_tok_length, tokenized_datasets_1):
    dic = {}
    for _sample in tokenized_datasets_1["train"]:
        sample_length = len(_sample["input_ids"])
        dic[sample_length] = dic.get(sample_length, 0) + 1
    for _i in range(1, max_tok_length + 1):
        if _i in dic:
            print(f"{_i:>2} {dic[_i]:>6}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Una muestra tras el filtrado:
    """)
    return


@app.cell
def _(tokenized_datasets_1):
    for _sample in tokenized_datasets_1["train"].select(range(5)):
        print(_sample["input_ids"])
        print(_sample["attention_mask"])
        print(_sample["labels"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Modelo cuantizado

    bitsandbytes es una librería de cuantización integrada en Transformers. Con la clase `BitsAndBytesConfig` podemos, por ejemplo:

    <ul>
    <li>usar load_in_4bit=True para cuantizar el modelo a 4 bits al cargarlo</li>
    <li>usar bnb_4bit_quant_type="nf4" para un tipo de dato de 4 bits pensado para pesos inicializados con una distribución normal</li>
    <li>usar bnb_4bit_use_double_quant=True para cuantizar también las constantes de cuantización</li>
    <li>usar bnb_4bit_compute_dtype=torch.bfloat16 para calcular en bfloat16, que es más rápido</li>
    </ul>
    """)
    return


@app.cell
def _():
    import torch
    from transformers import BitsAndBytesConfig

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    return quantization_config, torch


@app.cell
def _(checkpoint, quantization_config):
    from transformers import AutoModelForSeq2SeqLM

    model = AutoModelForSeq2SeqLM.from_pretrained(
        checkpoint,
        quantization_config=quantization_config,
    )
    return (model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Evaluación

    Medimos BLEU ([sacreBLEU](https://huggingface.co/spaces/evaluate-metric/sacrebleu)) y [COMET](https://huggingface.co/spaces/evaluate-metric/comet), las dos métricas que pide la práctica. COMET usa también el texto de entrada: aquí es el texto informal original.
    """)
    return


@app.cell
def _():
    from evaluate import load

    bleu_metric = load("sacrebleu")
    comet_metric = load("comet")
    return bleu_metric, comet_metric


@app.cell
def _(bleu_metric, comet_metric, np, tokenizer):
    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]
        return preds, labels

    def compute_metrics(eval_preds, sources):
        preds, labels = eval_preds

        # Convertir a lista si viene de una columna de datasets
        if not isinstance(labels, list):
            labels = list(labels)

        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        # Sustituir los ids negativos de las etiquetas, que no se pueden decodificar
        labels = [
            [tokenizer.pad_token_id if j < 0 else j for j in label] for label in labels
        ]
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        bleu = bleu_metric.compute(
            predictions=decoded_preds, references=[[l] for l in decoded_labels]
        )
        comet = comet_metric.compute(
            predictions=decoded_preds, references=decoded_labels, sources=list(sources)
        )

        prediction_lens = [
            np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
        ]
        result = {
            "bleu": bleu["score"],
            "comet": comet["mean_score"],
            "gen_len": np.mean(prediction_lens),
        }
        return {k: round(float(v), 4) for k, v in result.items()}

    return (compute_metrics,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Referencia trivial: copiar la entrada

    Como origen y destino están en el mismo idioma, conviene saber qué puntuación se obtiene sin modelo: devolver el texto informal tal cual. Un modelo útil debería superar esta cifra.
    """)
    return


@app.cell
def _(compute_metrics, tokenized_datasets_1):
    _test = tokenized_datasets_1["test"]
    copy_result = compute_metrics(
        (list(_test["input_ids"]), _test["labels"]), _test["original_text"]
    )
    print(f"Copiar la entrada -> BLEU: {copy_result['bleu']}  COMET: {copy_result['comet']}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Inferencia

    En inferencia usamos la [función generate](https://huggingface.co/docs/transformers/main_classes/text_generation), que codifica la entrada y genera la salida del decodificador de forma autorregresiva. Primero cargamos los parámetros de inferencia por defecto de NLLB:
    """)
    return


@app.cell
def _(checkpoint):
    from transformers import GenerationConfig

    generation_config = GenerationConfig.from_pretrained(checkpoint)

    print(generation_config)
    return (generation_config,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Preparamos el test en lotes:
    """)
    return


@app.cell
def _(tokenized_datasets_1):
    test_batch_size = 32
    batch_tokenized_test = tokenized_datasets_1["test"].batch(test_batch_size)
    return (batch_tokenized_test,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Procesamos cada lote (padding y conversión a tensores) y generamos con num_beams = 1 y do_sample = False, es decir, búsqueda voraz.
    """)
    return


@app.cell
def _(
    batch_tokenized_test,
    generation_config,
    max_tok_length,
    model,
    tgt_code,
    tokenizer,
    torch,
):
    number_of_batches = len(batch_tokenized_test["original_text"])
    output_sequences = []
    for _i in range(number_of_batches):
        inputs = tokenizer(
            batch_tokenized_test["original_text"][_i],
            max_length=max_tok_length,
            truncation=True,
            return_tensors="pt",
            padding=True,
        )
        with torch.no_grad():
            output_batch = model.generate(
                generation_config=generation_config,
                input_ids=inputs["input_ids"].cuda(),
                attention_mask=inputs["attention_mask"].cuda(),
                forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_code),
                max_length=max_tok_length,
                num_beams=1,
                do_sample=False,
            )
        output_sequences.extend(output_batch.cpu())
    return (output_sequences,)


@app.cell
def _(compute_metrics, output_sequences, tokenized_datasets_1):
    _test = tokenized_datasets_1["test"]
    result = compute_metrics((output_sequences, _test["labels"]), _test["original_text"])
    print(f"BLEU: {result['bleu']}  COMET: {result['comet']}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Algunos ejemplos del test: entrada informal, salida de NLLB y referencia formal.
    """)
    return


@app.cell
def _(mo, output_sequences, tokenized_datasets_1, tokenizer):
    _test = tokenized_datasets_1["test"].select(range(10))
    mo.ui.table(
        [
            {"original": o, "nllb": p, "formal": f}
            for o, p, f in zip(
                _test["original_text"],
                tokenizer.batch_decode(output_sequences[:10], skip_special_tokens=True),
                _test["formal_text"],
            )
        ],
        selection=None,
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
