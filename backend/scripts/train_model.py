import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.src.data.loader import load_processed_data
from backend.src.config import THRESHOLD_MIN_PRECISION
from backend.src.models.trainer import train_save_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["logreg", "random_forest"],
        default="logreg",
        help="Tipo de modelo para treino",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Nome do artefato salvo (sem extensão).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proporção do conjunto de teste.",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.2,
        help="Proporção do conjunto de validação.",
    )
    parser.add_argument(
        "--min-precision",
        type=float,
        default=THRESHOLD_MIN_PRECISION,
        help="Precisão mínima para seleção de threshold.",
    )
    args = parser.parse_args()

    print("Carregando dados processados...")
    df = load_processed_data("train_data.csv")
    print(f"Dados carregados: {df.shape}")

    print("Treinando modelo baseline...")
    model_name = args.name or ("baseline_logreg" if args.model == "logreg" else "rf_v1")
    result = train_save_report(
        df,
        model_name=model_name,
        model_type=args.model,
        test_size=args.test_size,
        val_size=args.val_size,
        threshold_min_precision=args.min_precision,
    )

    metrics = result["metrics"]
    print("\nResultados (validação):")
    for key, value in metrics["val"].items():
        print(f"- {key}: {value}")

    print("\nResultados (teste - threshold 0.5):")
    for key, value in metrics["test"].items():
        print(f"- {key}: {value}")

    if "best_threshold" in metrics:
        bt = metrics["best_threshold"]
        print(
            "\nMelhor threshold (priorizando recall): "
            f"thr={bt['threshold']}, recall={bt['recall']}, "
            f"precision={bt['precision']}, f1={bt['f1']}"
        )
        test_thr = metrics.get("test_at_threshold", {})
        if test_thr:
            print("\nResultados (teste - threshold escolhido):")
            for key, value in test_thr.items():
                print(f"- {key}: {value}")

    paths = result["paths"]
    print(f"\nModelo salvo em: {paths['model']}")
    print(f"Métricas salvas em: {paths['metrics']}")


if __name__ == "__main__":
    main()
