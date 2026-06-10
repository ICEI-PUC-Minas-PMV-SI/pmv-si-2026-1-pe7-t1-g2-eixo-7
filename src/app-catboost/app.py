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
    uvicorn app:app --reload
    # abra http://127.0.0.1:8000/docs  (interface para testar)
"""

from pathlib import Path
import json
import pickle

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

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

app = FastAPI(
    title="API de Previsão de Inadimplência",
    description=(
        "Recebe os dados de um pedido de empréstimo e devolve a probabilidade "
        "de inadimplência, usando o pipeline CatBoost treinado no notebook."
    ),
    version="1.0.0",
)


class Solicitacao(BaseModel):
    # ----- numéricas -----
    ID: int
    year: int
    loan_amount: float
    term: float
    income: float
    Credit_Score: int
    dtir1: float
    # ----- categóricas (texto) -----
    loan_limit: str
    Gender: str
    approv_in_adv: str
    loan_type: str
    loan_purpose: str
    Credit_Worthiness: str
    open_credit: str
    business_or_commercial: str
    Neg_ammortization: str
    interest_only: str
    lump_sum_payment: str
    construction_type: str
    occupancy_type: str
    Secured_by: str
    total_units: str
    credit_type: str
    # a coluna original tem hífen, que não é um nome válido em Python,
    # então usamos um alias: no JSON envie "co-applicant_credit_type".
    co_applicant_credit_type: str = Field(alias="co-applicant_credit_type")
    age: str
    submission_of_application: str
    Region: str
    Security_Type: str

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
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
        },
    )


class Resposta(BaseModel):
    probabilidade_inadimplencia: float  
    classe: int                         
    rotulo: str                         
    limiar: float                       


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
def raiz():
    return {"mensagem": "API de inadimplência no ar. Acesse /docs para testar."}


@app.get("/health")
def health():
    return {"status": "ok", "n_features": len(FEATURE_ORDER)}


@app.post("/prever", response_model=Resposta)
def prever(solicitacao: Solicitacao, limiar: float = Query(0.5, ge=0.0, le=1.0)):
    dados = solicitacao.model_dump(by_alias=True)

    df = pd.DataFrame([dados])

    for col in COLUNAS_CATEGORICAS:
        df[col] = df[col].astype(str).fillna("missing")

    df = df[FEATURE_ORDER]

    try:
        prob = float(modelo.predict_proba(df)[:, 1][0])  # probabilidade de Status=1
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Erro ao prever: {exc}")

    classe = int(prob >= limiar)
    return Resposta(
        probabilidade_inadimplencia=round(prob, 4),
        classe=classe,
        rotulo="Inadimplente" if classe == 1 else "Adimplente",
        limiar=limiar,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)