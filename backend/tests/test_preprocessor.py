import pandas as pd

from backend.src.data.preprocessor import preprocess_data


def test_preprocess_filters_defasagem_and_creates_target():
    df = pd.DataFrame(
        {
            "DEFASAGEM_2021": [1.0, -1.0, None],
            "INDE_2020": [6.0, 4.0, 5.0],
        }
    )

    processed = preprocess_data(df)

    assert processed.shape[0] == 2
    assert "RISCO_DEFASAGEM" in processed.columns
    assert processed["RISCO_DEFASAGEM"].tolist() == [0, 1]
