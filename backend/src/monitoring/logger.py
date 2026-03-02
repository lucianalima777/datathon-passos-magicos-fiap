import json

from loguru import logger

from backend.src.config import LOG_DIR

LOG_DIR.mkdir(parents=True, exist_ok=True)

logger.add(
    LOG_DIR / "predictions.log",
    rotation="10 MB",
    retention="30 days",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    serialize=True,
)


def log_prediction(
    input_data: dict, prediction: int, probability: float, time_ms: float
) -> None:
    """Registra predição com log estruturado."""
    logger.info(
        "predicao_realizada",
        input_hash=hash(json.dumps(input_data, sort_keys=True)),
        prediction=prediction,
        probability=round(probability, 4),
        response_time_ms=round(time_ms, 2),
    )
