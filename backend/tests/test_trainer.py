import pandas as pd

from backend.src.models.trainer import train_and_evaluate, save_artifacts


def _sample_df():
    return pd.DataFrame(
        {
            "INDE_2020": [5, 6, 4, 7, 3, 6, 5, 7],
            "IAA_2020": [5, 6, 4, 7, 3, 6, 5, 7],
            "PEDRA_2020_NORM": [
                "agata",
                "quartzo",
                "agata",
                "ametista",
                "quartzo",
                "topazio",
                "agata",
                "ametista",
            ],
            "RISCO_DEFASAGEM": [0, 1, 0, 1, 0, 1, 0, 1],
        }
    )


def test_train_and_evaluate_returns_metrics():
    df = _sample_df()
    model, metrics = train_and_evaluate(
        df, test_size=0.25, val_size=0.25, model_type="logreg"
    )

    assert "test" in metrics
    assert "val" in metrics
    assert "best_threshold" in metrics
    assert "test_at_threshold" in metrics
    assert metrics["model_type"] == "logreg"
    assert hasattr(model, "predict")


def test_save_artifacts_writes_files(tmp_path):
    df = _sample_df()
    model, metrics = train_and_evaluate(df, test_size=0.25, model_type="logreg")
    paths = save_artifacts(model, metrics, model_name="tmp_model", output_dir=tmp_path)

    assert paths["model"].exists()
    assert paths["metrics"].exists()
