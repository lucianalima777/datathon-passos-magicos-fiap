import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "backend" / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"

RAW_DATA_PATH = (
    BASE_DIR.parent
    / "dados-datathon"
    / "DATATHON"
    / "Bases antigas"
    / "PEDE_PASSOS_DATASET_FIAP.csv"
)

BASE_YEAR = 2020
TARGET_YEAR = 2021
DEFASAGEM_COL = f"DEFASAGEM_{TARGET_YEAR}"

TARGET_COLUMN = "RISCO_DEFASAGEM"
RANDOM_STATE = 42
THRESHOLD_MIN_PRECISION = 0.7
THRESHOLD_GRID = [
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
]

INDICADORES = ["INDE", "IAA", "IEG", "IPS", "IDA", "IPP", "IPV", "IAN"]


def year_cols(prefixes: list[str], year: int) -> list[str]:
    return [f"{prefix}_{year}" for prefix in prefixes]


def indicadores_year(year: int = BASE_YEAR) -> list[str]:
    return year_cols(INDICADORES, year)


COLUNAS_NUMERICAS = [
    "IDADE_ALUNO_2020",
    "ANOS_PM_2020",
    "INDE_2020",
    "IAA_2020",
    "IEG_2020",
    "IPS_2020",
    "IDA_2020",
    "IPP_2020",
    "IPV_2020",
    "IAN_2020",
    "INDE_2021",
    "IAA_2021",
    "IEG_2021",
    "IPS_2021",
    "IDA_2021",
    "IPP_2021",
    "IPV_2021",
    "IAN_2021",
    DEFASAGEM_COL,
    "INDE_2022",
    "IAA_2022",
    "IEG_2022",
    "IPS_2022",
    "IDA_2022",
    "IPP_2022",
    "IPV_2022",
    "IAN_2022",
    "NOTA_PORT_2022",
    "NOTA_MAT_2022",
    "NOTA_ING_2022",
    "CG_2022",
    "CF_2022",
    "CT_2022",
    "QTD_AVAL_2022",
    "ANO_INGRESSO_2022",
]

FEATURES_CATEGORICAS = [
    f"PEDRA_{BASE_YEAR}_NORM",
    f"PONTO_VIRADA_{BASE_YEAR}_NORM",
    f"INSTITUICAO_ENSINO_ALUNO_{BASE_YEAR}_NORM",
    f"FASE_TURMA_{BASE_YEAR}_NORM",
]

MODEL_NAME = "baseline_logreg"
DB_PATH = DATA_DIR / "predictions.db"
LOG_DIR = BASE_DIR / "backend" / "logs"
API_HOST = "0.0.0.0"
API_PORT = 8000
DRIFT_MIN_SAMPLES = 30

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"
