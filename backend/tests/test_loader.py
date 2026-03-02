import pandas as pd
import pytest

from backend.src.data.loader import load_processed_data, load_raw_data, save_processed_data


def test_load_raw_data_com_caminho_customizado(tmp_path):
    csv_file = tmp_path / "raw.csv"
    csv_file.write_text("NOME;INDE_2020\nALUNO-1;7.5\nALUNO-2;6.0", encoding="utf-8")

    df = load_raw_data(file_path=csv_file)

    assert len(df) == 2
    assert "INDE_2020" in df.columns
    assert "NOME" in df.columns


def test_save_e_load_processed_data(tmp_path, monkeypatch):
    import backend.src.config as config

    monkeypatch.setattr(config, "PROCESSED_DATA_DIR", tmp_path)

    df_original = pd.DataFrame({"feature_a": [1.0, 2.0], "target": [0, 1]})
    save_processed_data(df_original, "teste.csv")

    df_carregado = load_processed_data("teste.csv")

    assert len(df_carregado) == 2
    assert list(df_carregado.columns) == ["feature_a", "target"]


def test_load_processed_data_arquivo_inexistente(tmp_path, monkeypatch):
    import backend.src.config as config

    monkeypatch.setattr(config, "PROCESSED_DATA_DIR", tmp_path)

    with pytest.raises(FileNotFoundError):
        load_processed_data("nao_existe.csv")
