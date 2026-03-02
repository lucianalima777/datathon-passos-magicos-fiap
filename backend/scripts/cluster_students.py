"""Script para treinar o modelo de clusterização de perfis de alunos.

Uso:
    python backend/scripts/cluster_students.py

Gera o artefato kmeans_profiles.joblib em backend/data/models/.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.src.data.loader import load_processed_data
from backend.src.models.clusterer import CLUSTER_COLS, N_CLUSTERS, train_cluster_model


def main():
    print("Carregando dados processados...")
    df = load_processed_data("train_data.csv")
    print(f"Dados carregados: {df.shape}")

    available = [c for c in CLUSTER_COLS if c in df.columns]
    print(f"Colunas para clustering ({len(available)}/{len(CLUSTER_COLS)}): {available}")

    n_validas = df[available].dropna().shape[0]
    print(f"Amostras válidas (sem NaN nos indicadores): {n_validas}")

    print(f"\nTreinando K-Means com {N_CLUSTERS} clusters...")
    result = train_cluster_model(df)

    print(f"\nModelo salvo em: {result['path']}")
    print(f"Inercia: {result['inertia']}")
    print("\nDistribuicao dos clusters:")
    for label, count in result["cluster_sizes"].items():
        print(f"  {label}: {count} alunos")


if __name__ == "__main__":
    main()
