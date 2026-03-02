from typing import Optional

from pydantic import BaseModel, Field


class PredictionInput(BaseModel):
    """Dados brutos do aluno para predição de risco de defasagem."""

    INDE_2020: float = Field(..., ge=0.0, le=10.0, description="Índice de Desenvolvimento Educacional")
    IAA_2020: float = Field(..., ge=0.0, le=10.0, description="Indicador de Autoavaliação")
    IEG_2020: float = Field(..., ge=0.0, le=10.0, description="Indicador de Engajamento")
    IPS_2020: float = Field(..., ge=0.0, le=10.0, description="Indicador Psicossocial")
    IDA_2020: float = Field(..., ge=0.0, le=10.0, description="Indicador de Aprendizagem")
    IPP_2020: float = Field(..., ge=0.0, le=10.0, description="Indicador Psicopedagógico")
    IPV_2020: float = Field(..., ge=0.0, le=10.0, description="Indicador de Ponto de Virada")
    IAN_2020: float = Field(..., ge=0.0, le=10.0, description="Indicador de Adequação de Nível")
    IDADE_ALUNO_2020: int = Field(..., ge=5, le=25, description="Idade do aluno")
    ANOS_PM_2020: int = Field(..., ge=0, le=15, description="Anos na Passos Mágicos")
    PEDRA_2020: str = Field(..., description="Classificação por pedra")
    PONTO_VIRADA_2020: str = Field(..., description="Se atingiu ponto de virada")
    INSTITUICAO_ENSINO_ALUNO_2020: str = Field(..., description="Instituição de ensino")
    FASE_TURMA_2020: Optional[str] = Field(
        None, description="Fase e turma do aluno (ex: Fase 1 - Turma A)"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "INDE_2020": 6.5,
                    "IAA_2020": 6.0,
                    "IEG_2020": 5.0,
                    "IPS_2020": 6.0,
                    "IDA_2020": 6.0,
                    "IPP_2020": 6.0,
                    "IPV_2020": 5.0,
                    "IAN_2020": 6.0,
                    "IDADE_ALUNO_2020": 12,
                    "ANOS_PM_2020": 2,
                    "PEDRA_2020": "Agata",
                    "PONTO_VIRADA_2020": "Sim",
                    "INSTITUICAO_ENSINO_ALUNO_2020": "Publica",
                    "FASE_TURMA_2020": "Fase 1 - Turma A",
                }
            ]
        }
    }


class PredictionOutput(BaseModel):
    """Resultado da predição de risco de defasagem."""

    risco_defasagem: int = Field(..., description="1=em risco, 0=adequado")
    probabilidade: float = Field(..., description="Probabilidade de risco (0-1)")
    threshold: float = Field(..., description="Threshold utilizado")
    classificacao: str = Field(..., description="Em Risco / Adequado")


class HealthOutput(BaseModel):
    """Status da API."""

    model_config = {"protected_namespaces": ()}

    status: str
    model_name: str
    model_version: str
    total_predictions: int


class ExplainOutput(BaseModel):
    """Resultado da analise pedagogica via LLM."""

    relatorio: str = Field(..., description="Relatorio pedagogico gerado pelo assistente")
    risco_defasagem: int = Field(..., description="1=em risco, 0=adequado")
    probabilidade: float = Field(..., description="Probabilidade de risco (0-1)")
    classificacao: str = Field(..., description="Em Risco / Adequado")


class ClusterOutput(BaseModel):
    """Resultado da clusterização por perfil pedagógico."""

    cluster_id: int = Field(..., description="ID numérico do cluster (0-3)")
    perfil: str = Field(..., description="Label interpretável do perfil")
    distancias: dict = Field(..., description="Distância euclidiana para cada centroide")
    descricao: str = Field(..., description="Descrição pedagógica do perfil")
