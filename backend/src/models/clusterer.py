"""Módulo de clusterização de alunos por perfil de indicadores.

Usa K-Means sobre os 8 indicadores normalizados do ano base para
agrupar alunos em perfis pedagógicos interpretáveis.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from backend.src.config import BASE_YEAR, INDICADORES, MODELS_DIR, RANDOM_STATE

CLUSTER_COLS = [f"{ind}_{BASE_YEAR}" for ind in INDICADORES]

N_CLUSTERS = 4

CLUSTER_LABELS_BY_RANK = [
    "risco_alto",
    "risco_moderado",
    "em_desenvolvimento",
    "alto_desempenho",
]

CLUSTER_DESCRICOES: dict[str, str] = {
    "risco_alto": (
        "Aluno com desempenho baixo generalizado nos indicadores. "
        "Necessita de intervencao pedagogica prioritaria e acompanhamento intensivo."
    ),
    "risco_moderado": (
        "Aluno com indicadores mistos, apresentando fragilidades em areas especificas. "
        "Requer acompanhamento direcionado para as dimensoes mais defasadas."
    ),
    "em_desenvolvimento": (
        "Aluno em trajetoria positiva, com indicadores acima da media. "
        "Pode se beneficiar de desafios adicionais para consolidar o progresso."
    ),
    "alto_desempenho": (
        "Aluno com desempenho elevado em todos os indicadores. "
        "Apresenta base solida para progressao e pode atuar como lider de turma."
    ),
}


def _assign_labels(kmeans: KMeans, scaler: StandardScaler) -> dict[int, str]:
    centroides = scaler.inverse_transform(kmeans.cluster_centers_)
    medias = centroides.mean(axis=1)
    ranking = np.argsort(medias)
    return {int(cluster_id): CLUSTER_LABELS_BY_RANK[rank] for rank, cluster_id in enumerate(ranking)}


def train_cluster_model(df: pd.DataFrame) -> dict:
    available = [c for c in CLUSTER_COLS if c in df.columns]
    if len(available) < len(CLUSTER_COLS):
        missing = set(CLUSTER_COLS) - set(available)
        raise ValueError(f"Colunas ausentes para clustering: {missing}")

    X = df[CLUSTER_COLS].dropna()
    if len(X) < N_CLUSTERS:
        raise ValueError(f"Dados insuficientes: {len(X)} amostras para {N_CLUSTERS} clusters.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
    kmeans.fit(X_scaled)

    label_map = _assign_labels(kmeans, scaler)

    artifact = {"kmeans": kmeans, "scaler": scaler, "label_map": label_map}
    artifact_path = MODELS_DIR / "kmeans_profiles.joblib"
    joblib.dump(artifact, artifact_path)

    inertia = float(kmeans.inertia_)
    sizes = {label_map[i]: int((kmeans.labels_ == i).sum()) for i in range(N_CLUSTERS)}

    return {
        "path": str(artifact_path),
        "n_clusters": N_CLUSTERS,
        "inertia": round(inertia, 2),
        "cluster_sizes": sizes,
    }


def load_cluster_model() -> dict:
    artifact_path = MODELS_DIR / "kmeans_profiles.joblib"
    if not artifact_path.exists():
        raise RuntimeError(
            "Modelo de clusterizacao nao encontrado. "
            "Execute: python backend/scripts/cluster_students.py"
        )
    return joblib.load(artifact_path)


def predict_cluster(artifact: dict, input_df: pd.DataFrame) -> tuple[int, str, dict]:
    kmeans: KMeans = artifact["kmeans"]
    scaler: StandardScaler = artifact["scaler"]
    label_map: dict[int, str] = artifact["label_map"]

    available = [c for c in CLUSTER_COLS if c in input_df.columns]
    X = input_df[available].fillna(0).reindex(columns=CLUSTER_COLS, fill_value=0)

    X_scaled = scaler.transform(X)
    cluster_id = int(kmeans.predict(X_scaled)[0])
    label = label_map[cluster_id]

    distances = kmeans.transform(X_scaled)[0]
    dist_by_label = {label_map[i]: round(float(distances[i]), 4) for i in range(N_CLUSTERS)}

    return cluster_id, label, dist_by_label
