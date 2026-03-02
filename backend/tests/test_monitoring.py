import pytest

from backend.src.monitoring.drift import NUMERIC_DRIFT_COLS, check_data_drift


def test_drift_detection_with_same_distribution():
    import pandas as pd

    from backend.src.config import PROCESSED_DATA_DIR

    ref_path = PROCESSED_DATA_DIR / "train_data.csv"
    if not ref_path.exists():
        pytest.skip("train_data.csv nao encontrado")

    ref = pd.read_csv(ref_path)
    available = [c for c in NUMERIC_DRIFT_COLS if c in ref.columns]
    reference = ref[available]
    sample = reference.head(50).to_dict("records")

    result = check_data_drift(sample, reference)
    assert result["status"] == "completed"
    assert "dataset_drift" in result
    assert "drift_share" in result
    assert "n_drifted_columns" in result
    assert isinstance(result["column_details"], dict)


def test_drift_columns_match():
    assert len(NUMERIC_DRIFT_COLS) == 8
    assert "INDE_2020" in NUMERIC_DRIFT_COLS
