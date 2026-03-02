import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


class PredictionStore:
    """Armazena predições em SQLite para histórico e monitoring."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    input_data TEXT NOT NULL,
                    prediction INTEGER NOT NULL,
                    probability REAL NOT NULL,
                    threshold REAL NOT NULL,
                    model_name TEXT NOT NULL,
                    response_time_ms REAL
                )
            """)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path), check_same_thread=False)

    def save_prediction(
        self,
        input_data: dict,
        prediction: int,
        probability: float,
        threshold: float,
        model_name: str,
        response_time_ms: float | None = None,
    ) -> int:
        """Insere predição e retorna o ID."""
        ts = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO predictions
                    (timestamp, input_data, prediction, probability,
                     threshold, model_name, response_time_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts,
                    json.dumps(input_data),
                    prediction,
                    probability,
                    threshold,
                    model_name,
                    response_time_ms,
                ),
            )
            return cursor.lastrowid

    def get_predictions(
        self, limit: int = 100, prediction: int | None = None
    ) -> list[dict]:
        """Consulta histórico de predições."""
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            if prediction is not None:
                rows = conn.execute(
                    "SELECT * FROM predictions WHERE prediction = ? ORDER BY id DESC LIMIT ?",
                    (prediction, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM predictions ORDER BY id DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        return [dict(row) for row in rows]

    def get_recent_inputs(self, n: int = 50) -> list[dict]:
        """Retorna inputs recentes para drift detection."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT input_data FROM predictions ORDER BY id DESC LIMIT ?",
                (n,),
            ).fetchall()
        return [json.loads(row[0]) for row in rows]

    def count(self) -> int:
        """Retorna total de predições armazenadas."""
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
