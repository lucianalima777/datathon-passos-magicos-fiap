from __future__ import annotations

import hashlib
import json
import time
from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from loguru import logger

from backend.src.api.schemas import ClusterOutput, ExplainOutput, HealthOutput, PredictionInput, PredictionOutput
from backend.src.config import (
    DB_PATH,
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    DRIFT_MIN_SAMPLES,
    MODEL_NAME,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
)
from backend.src.features.engineer import engineer_features
from backend.src.models.clusterer import CLUSTER_DESCRICOES, load_cluster_model, predict_cluster
from backend.src.monitoring.drift import NUMERIC_DRIFT_COLS, check_data_drift
from backend.src.monitoring.logger import log_prediction
from backend.src.monitoring.store import PredictionStore


def _load_threshold(model_name: str) -> float:
    metrics_path = MODELS_DIR / f"{model_name}_metrics.json"
    if not metrics_path.exists():
        raise RuntimeError(
            "Artefato de métricas não encontrado. "
            "Gere e disponibilize o arquivo "
            f"{metrics_path.name} em {metrics_path.parent}."
        )
    with open(metrics_path) as f:
        metrics = json.load(f)
    return metrics["best_threshold"]["threshold"]


def _load_metrics(model_name: str) -> dict:
    metrics_path = MODELS_DIR / f"{model_name}_metrics.json"
    with open(metrics_path) as f:
        return json.load(f)


def _compute_model_hash(model_path) -> str:
    return hashlib.sha256(model_path.read_bytes()).hexdigest()[:12]


def _load_drift_reference() -> pd.DataFrame | None:
    ref_path = PROCESSED_DATA_DIR / "train_data.csv"
    if not ref_path.exists():
        logger.warning("Referencia de drift nao encontrada: {}", ref_path)
        return None
    df = pd.read_csv(ref_path)
    available = [c for c in NUMERIC_DRIFT_COLS if c in df.columns]
    return df[available]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carrega modelo, threshold, métricas e referência de drift no startup."""
    model_path = MODELS_DIR / f"{MODEL_NAME}.joblib"
    if not model_path.exists():
        raise RuntimeError(
            "Modelo não encontrado. "
            "Gere e disponibilize o arquivo "
            f"{model_path.name} em {model_path.parent}."
        )
    app.state.model = joblib.load(model_path)
    app.state.threshold = _load_threshold(MODEL_NAME)
    app.state.model_name = MODEL_NAME
    app.state.model_version = _compute_model_hash(model_path)
    app.state.model_metrics = _load_metrics(MODEL_NAME)
    app.state.store = PredictionStore(DB_PATH)
    app.state.drift_reference = _load_drift_reference()
    try:
        app.state.cluster_artifact = load_cluster_model()
        logger.info("Modelo de clusterizacao carregado.")
    except RuntimeError as e:
        logger.warning("Modelo de clusterizacao indisponivel: {}. Endpoint /cluster desativado.", e)
        app.state.cluster_artifact = None
    yield


app = FastAPI(
    title="Passos Mágicos - Predição de Risco",
    description="API de predição de risco de defasagem escolar",
    lifespan=lifespan,
)


def prepare_input_for_model(input_data: PredictionInput) -> pd.DataFrame:
    """Converte input da API em DataFrame com features engenheiradas."""
    row = input_data.model_dump()
    df = pd.DataFrame([row])
    df_engineered = engineer_features(df)
    return df_engineered.drop(columns=["RISCO_DEFASAGEM"], errors="ignore")


@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput, request: Request):
    """Realiza predição de risco de defasagem."""
    start = time.perf_counter()

    X = prepare_input_for_model(input_data)
    proba = float(request.app.state.model.predict_proba(X)[:, 1][0])
    threshold = request.app.state.threshold
    prediction = int(proba >= threshold)

    elapsed_ms = (time.perf_counter() - start) * 1000

    input_dict = input_data.model_dump()
    try:
        request.app.state.store.save_prediction(
            input_data=input_dict,
            prediction=prediction,
            probability=proba,
            threshold=threshold,
            model_name=request.app.state.model_name,
            response_time_ms=elapsed_ms,
        )
    except Exception as e:
        logger.warning("Falha ao persistir predicao no banco: {}", e)

    try:
        log_prediction(input_dict, prediction, proba, elapsed_ms)
    except Exception as e:
        logger.warning("Falha ao registrar log da predicao: {}", e)

    return PredictionOutput(
        risco_defasagem=prediction,
        probabilidade=round(proba, 4),
        threshold=round(threshold, 4),
        classificacao="Em Risco" if prediction == 1 else "Adequado",
    )


@app.get("/health", response_model=HealthOutput)
async def health(request: Request):
    """Verifica status da API."""
    store: PredictionStore = request.app.state.store
    return HealthOutput(
        status="ok",
        model_name=request.app.state.model_name,
        model_version=request.app.state.model_version,
        total_predictions=store.count(),
    )


@app.get("/metrics")
async def get_model_metrics(request: Request):
    """Retorna métricas de treinamento e validação do modelo carregado."""
    return request.app.state.model_metrics


@app.get("/predictions")
async def get_predictions(
    request: Request,
    limit: int = Query(50, ge=1, le=500),
    risco: int | None = Query(None, ge=0, le=1),
):
    """Retorna histórico de predições."""
    store: PredictionStore = request.app.state.store
    return store.get_predictions(limit=limit, prediction=risco)


def _build_explain_prompt(data: PredictionInput, classificacao: str, probabilidade: float) -> str:
    return (
        "Você é um assistente pedagógico especialista da Associação Passos Mágicos, "
        "uma ONG de transformação social que apoia crianças e jovens em situação de vulnerabilidade "
        "através da educação de qualidade.\n\n"
        "DADOS DO ALUNO:\n"
        f"- INDE (Desenvolvimento Educacional): {data.INDE_2020:.1f}/10\n"
        f"- IAA (Autoavaliação): {data.IAA_2020:.1f}/10\n"
        f"- IEG (Engajamento): {data.IEG_2020:.1f}/10\n"
        f"- IPS (Psicossocial): {data.IPS_2020:.1f}/10\n"
        f"- IDA (Aprendizagem): {data.IDA_2020:.1f}/10\n"
        f"- IPP (Psicopedagógico): {data.IPP_2020:.1f}/10\n"
        f"- IPV (Ponto de Virada): {data.IPV_2020:.1f}/10\n"
        f"- IAN (Adequação ao Nível): {data.IAN_2020:.1f}/10\n"
        f"- Idade: {data.IDADE_ALUNO_2020} anos | Anos na Passos Mágicos: {data.ANOS_PM_2020}\n"
        f"- Pedra: {data.PEDRA_2020} | Ponto de Virada: {data.PONTO_VIRADA_2020}\n"
        f"- Instituição: {data.INSTITUICAO_ENSINO_ALUNO_2020}\n\n"
        "AVALIAÇÃO PREDITIVA:\n"
        f"- Status: {classificacao}\n"
        f"- Probabilidade de risco de defasagem: {probabilidade:.0%}\n\n"
        "Você está respondendo a um professor ou coordenador pedagógico da Passos Mágicos.\n"
        "Use linguagem empática, prática e acessível. Evite jargões técnicos de machine learning."
    )


@app.post("/explain", response_model=ExplainOutput)
async def explain_student(input_data: PredictionInput, request: Request):
    """Gera relatorio pedagogico personalizado via LLM para o aluno informado."""
    if not DEEPSEEK_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="Assistente pedagogico nao configurado. Defina DEEPSEEK_API_KEY no ambiente.",
        )

    X = prepare_input_for_model(input_data)
    proba = float(request.app.state.model.predict_proba(X)[:, 1][0])
    threshold = request.app.state.threshold
    prediction = int(proba >= threshold)
    classificacao = "Em Risco" if prediction == 1 else "Adequado"

    system_prompt = _build_explain_prompt(input_data, classificacao, proba)
    user_msg = (
        "Gere um relatório pedagógico personalizado contendo:\n"
        "1. Pontos fortes do aluno (2-3 linhas)\n"
        "2. Principais áreas de atenção (2-3 linhas)\n"
        "3. Três recomendações práticas de intervenção, em ordem de prioridade\n\n"
        "Seja direto e empático. Máximo 300 palavras."
    )

    from openai import OpenAI as OpenAIClient

    ai_client = OpenAIClient(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    try:
        response = ai_client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=600,
            temperature=0.7,
        )
        relatorio = response.choices[0].message.content
    except Exception as e:
        logger.warning("Falha ao chamar DeepSeek: {}", e)
        raise HTTPException(status_code=502, detail=f"Erro ao gerar relatorio: {e}")

    return ExplainOutput(
        relatorio=relatorio,
        risco_defasagem=prediction,
        probabilidade=round(proba, 4),
        classificacao=classificacao,
    )


@app.post("/cluster", response_model=ClusterOutput)
async def cluster_student(input_data: PredictionInput, request: Request):
    """Classifica o aluno em um perfil pedagogico via K-Means."""
    artifact = request.app.state.cluster_artifact
    if artifact is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo de clusterizacao nao disponivel. Execute cluster_students.py.",
        )

    row = input_data.model_dump()
    df = pd.DataFrame([row])
    cluster_id, label, distances = predict_cluster(artifact, df)
    descricao = CLUSTER_DESCRICOES.get(label, "Perfil nao identificado.")

    return ClusterOutput(
        cluster_id=cluster_id,
        perfil=label,
        distancias=distances,
        descricao=descricao,
    )


@app.get("/drift")
async def check_drift(
    request: Request,
    n: int = Query(50, ge=10, le=500),
):
    """Verifica drift nos inputs recentes vs dados de treino."""
    store: PredictionStore = request.app.state.store
    total = store.count()
    if total < DRIFT_MIN_SAMPLES:
        return {
            "status": "insufficient_data",
            "message": f"Necessário pelo menos {DRIFT_MIN_SAMPLES} predições. Atual: {total}",
        }

    reference = request.app.state.drift_reference
    if reference is None:
        return {
            "status": "reference_unavailable",
            "message": "Dados de referência para drift não disponíveis.",
        }

    recent_inputs = store.get_recent_inputs(n=n)
    result = check_data_drift(recent_inputs, reference)
    return result
