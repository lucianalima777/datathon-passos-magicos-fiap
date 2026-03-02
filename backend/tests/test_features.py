import pandas as pd

from backend.src.features.engineer import engineer_features


def test_engineer_features_normalizes_categoricals():
    df = pd.DataFrame(
        {
            "INDE_2020": [6.5, 4.2],
            "IAA_2020": [6.0, 4.0],
            "IEG_2020": [5.0, 3.0],
            "IPS_2020": [6.0, 4.0],
            "IDA_2020": [6.0, 3.0],
            "IPP_2020": [6.0, 4.0],
            "IPV_2020": [5.0, 2.0],
            "IAN_2020": [6.0, 3.0],
            "PEDRA_2020": ["Ágata", "Quartzo"],
            "PONTO_VIRADA_2020": ["Sim", "Não"],
            "INSTITUICAO_ENSINO_ALUNO_2020": ["Pública", "Privada"],
            "IDADE_ALUNO_2020": [12, 15],
            "ANOS_PM_2020": [2, 3],
            "RISCO_DEFASAGEM": [0, 1],
        }
    )

    features = engineer_features(df)

    assert "PEDRA_2020_NORM" in features.columns
    assert "PONTO_VIRADA_2020_NORM" in features.columns
    assert "INSTITUICAO_ENSINO_ALUNO_2020_NORM" in features.columns
    assert features["PEDRA_2020_NORM"].tolist() == ["agata", "quartzo"]
