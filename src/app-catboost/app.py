"""
API de Previsão de Inadimplência (CatBoost)
===========================================

Como funciona a comunicação backend <-> modelo (a mesma ideia do notebook BMI,
só que exposta via HTTP):

    1) Na SUBIDA do servidor o pipeline é carregado UMA vez do arquivo .pkl
       (no BMI: `pickle.load(file)`). Não recarregamos o modelo a cada requisição.
    2) O cliente manda um JSON com os dados do pedido de empréstimo.
    3) O backend transforma esse JSON num DataFrame de 1 linha
       (no BMI: `pd.DataFrame(config)`), passa pelo `pipeline.predict_proba`
       e devolve outro JSON com a decisão.

Ou seja: o `predict_mpg(config, model)` do BMI vira o endpoint POST /prever.

Para rodar:
    pip install -r requirements.txt
    flask --app app run --reload
    # ou simplesmente: python app.py
    # depois mande um POST para http://127.0.0.1:8000/prever
"""

from datetime import datetime
from pathlib import Path
import json
import pickle

import pandas as pd
from flask import Flask, jsonify, request

# ---------------------------------------------------------------------------
# 1) Carregamento do modelo + esquema (UMA vez, quando o servidor sobe)
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODELO_PATH = BASE_DIR / "modelo_inadimplencia.pkl"
SCHEMA_PATH = BASE_DIR / "colunas_modelo.json"

if not MODELO_PATH.exists() or not SCHEMA_PATH.exists():
    raise RuntimeError(
        "Arquivos do modelo não encontrados.")

with open(MODELO_PATH, "rb") as f:
    modelo = pickle.load(f)

with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
    schema = json.load(f)

FEATURE_ORDER = schema["feature_order"]
COLUNAS_CATEGORICAS = schema["colunas_categoricas"]

app = Flask(__name__)

# ---------------------------------------------------------------------------
# 2) Esquema de entrada (substitui o modelo Pydantic do FastAPI)
#
# Cada campo aponta para o tipo esperado. A coluna original
# "co-applicant_credit_type" tem hífen, então no JSON é enviada com esse nome
# mesmo — aqui não precisamos de alias porque trabalhamos direto com o dict.
# ---------------------------------------------------------------------------
# ID e year NÃO são preenchidos pelo cliente: o ID vai vazio e o year recebe
# o ano atual (preenchidos no servidor, ver `validar_e_converter`).
CAMPOS_NUMERICOS_INT = ["Credit_Score"]
CAMPOS_NUMERICOS_FLOAT = ["loan_amount", "term", "income", "dtir1"]
CAMPOS_TEXTO = [
    "loan_limit", "Gender", "approv_in_adv", "loan_type", "loan_purpose",
    "Credit_Worthiness", "open_credit", "business_or_commercial",
    "Neg_ammortization", "interest_only", "lump_sum_payment",
    "construction_type", "occupancy_type", "Secured_by", "total_units",
    "credit_type", "co-applicant_credit_type", "age",
    "submission_of_application", "Region", "Security_Type",
]

CAMPOS_OBRIGATORIOS = CAMPOS_NUMERICOS_INT + CAMPOS_NUMERICOS_FLOAT + CAMPOS_TEXTO

EXEMPLO = {
    "ID": 24890,
    "year": 2019,
    "loan_amount": 206500.0,
    "term": 360.0,
    "income": 4980.0,
    "Credit_Score": 758,
    "dtir1": 39.0,
    "loan_limit": "cf",
    "Gender": "Male",
    "approv_in_adv": "nopre",
    "loan_type": "type1",
    "loan_purpose": "p1",
    "Credit_Worthiness": "l1",
    "open_credit": "nopc",
    "business_or_commercial": "nob/c",
    "Neg_ammortization": "not_neg",
    "interest_only": "not_int",
    "lump_sum_payment": "not_lpsm",
    "construction_type": "sb",
    "occupancy_type": "pr",
    "Secured_by": "home",
    "total_units": "1U",
    "credit_type": "EXP",
    "co-applicant_credit_type": "CIB",
    "age": "45-54",
    "submission_of_application": "to_inst",
    "Region": "south",
    "Security_Type": "direct",
}


def validar_e_converter(dados):
    """Valida os campos recebidos e devolve um dict pronto para o DataFrame.

    Levanta ValueError com uma mensagem amigável quando algo está errado
    (campo faltando ou tipo inválido), papel que antes era do Pydantic.
    """
    faltando = [campo for campo in CAMPOS_OBRIGATORIOS if campo not in dados]
    if faltando:
        raise ValueError(f"Campos obrigatórios ausentes: {', '.join(faltando)}")

    convertido = {}
    for campo in CAMPOS_NUMERICOS_INT:
        try:
            convertido[campo] = int(dados[campo])
        except (TypeError, ValueError):
            raise ValueError(f"Campo '{campo}' deve ser um inteiro.")
    for campo in CAMPOS_NUMERICOS_FLOAT:
        try:
            convertido[campo] = float(dados[campo])
        except (TypeError, ValueError):
            raise ValueError(f"Campo '{campo}' deve ser um número.")
    for campo in CAMPOS_TEXTO:
        convertido[campo] = str(dados[campo])

    # Campos preenchidos pelo servidor (não vêm do formulário):
    convertido["ID"] = ""                     # ID vai vazio
    convertido["year"] = datetime.now().year  # ano atual (lib nativa)
    return convertido


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
def raiz():
    return jsonify(
        {"mensagem": "API de inadimplência no ar. Mande um POST para /prever."}
    )


@app.get("/ui")
def ui():
    # Entrega a página web do formulário (mesma origem da API → sem CORS).
    return app.send_static_file("index.html")


@app.get("/health")
def health():
    return jsonify({"status": "ok", "n_features": len(FEATURE_ORDER)})


@app.post("/prever")
def prever():
    corpo = request.get_json(silent=True)
    if not isinstance(corpo, dict):
        return jsonify({"detail": "Envie um corpo JSON válido."}), 400

    # limiar pode vir como query string (?limiar=0.5), igual ao FastAPI
    try:
        limiar = float(request.args.get("limiar", 0.5))
    except ValueError:
        return jsonify({"detail": "limiar deve ser um número."}), 400
    if not 0.0 <= limiar <= 1.0:
        return jsonify({"detail": "limiar deve estar entre 0.0 e 1.0."}), 400

    try:
        dados = validar_e_converter(corpo)
    except ValueError as exc:
        return jsonify({"detail": str(exc)}), 422

    df = pd.DataFrame([dados])

    for col in COLUNAS_CATEGORICAS:
        df[col] = df[col].astype(str).fillna("missing")

    df = df[FEATURE_ORDER]

    try:
        prob = float(modelo.predict_proba(df)[:, 1][0])  # probabilidade de Status=1
    except Exception as exc:  # noqa: BLE001
        return jsonify({"detail": f"Erro ao prever: {exc}"}), 500

    classe = int(prob >= limiar)
    return jsonify(
        {
            "probabilidade_inadimplencia": round(prob, 4),
            "classe": classe,
            "rotulo": "Inadimplente" if classe == 1 else "Adimplente",
            "limiar": limiar,
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
