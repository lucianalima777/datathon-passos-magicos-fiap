import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.src.api.schemas import PredictionInput
from backend.src.monitoring.store import PredictionStore


SAMPLE_INPUT_DICT = {
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
}


@pytest.fixture
def sample_input() -> dict:
    return SAMPLE_INPUT_DICT.copy()


@pytest.fixture
def sample_prediction_input() -> PredictionInput:
    return PredictionInput(**SAMPLE_INPUT_DICT)


@pytest.fixture
def tmp_db(tmp_path) -> Path:
    return tmp_path / "test_predictions.db"


@pytest.fixture
def prediction_store(tmp_db) -> PredictionStore:
    return PredictionStore(tmp_db)


@pytest.fixture
def sample_engineered_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "INDE_2020": [6.5, 4.2],
            "IAA_2020": [6.0, 4.0],
            "IEG_2020": [5.0, 3.0],
            "IPS_2020": [6.0, 4.0],
            "IDA_2020": [6.0, 3.0],
            "IPP_2020": [6.0, 4.0],
            "IPV_2020": [5.0, 2.0],
            "IAN_2020": [6.0, 3.0],
            "PEDRA_2020": ["Agata", "Quartzo"],
            "PONTO_VIRADA_2020": ["Sim", "Nao"],
            "INSTITUICAO_ENSINO_ALUNO_2020": ["Publica", "Privada"],
            "IDADE_ALUNO_2020": [12, 15],
            "ANOS_PM_2020": [2, 3],
            "RISCO_DEFASAGEM": [0, 1],
        }
    )
