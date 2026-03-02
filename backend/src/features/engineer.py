import unicodedata

import pandas as pd

from backend.src.config import BASE_YEAR


def create_basic_features(df: pd.DataFrame, base_year: int = BASE_YEAR) -> pd.DataFrame:
    df_copy = df.copy()

    indicadores = ["IAA", "IEG", "IPS", "IDA", "IPP", "IPV", "IAN"]
    indicadores = [f"{name}_{base_year}" for name in indicadores]

    valid_indicadores = [col for col in indicadores if col in df_copy.columns]

    if valid_indicadores:
        df_copy[f"MEDIA_INDICADORES_{base_year}"] = df_copy[valid_indicadores].mean(axis=1)
        df_copy[f"STD_INDICADORES_{base_year}"] = df_copy[valid_indicadores].std(axis=1)
        df_copy[f"MIN_INDICADORES_{base_year}"] = df_copy[valid_indicadores].min(axis=1)
        df_copy[f"MAX_INDICADORES_{base_year}"] = df_copy[valid_indicadores].max(axis=1)

    return df_copy


def create_ratio_features(df: pd.DataFrame, base_year: int = BASE_YEAR) -> pd.DataFrame:
    df_copy = df.copy()

    ida_col = f"IDA_{base_year}"
    ian_col = f"IAN_{base_year}"
    if ida_col in df_copy.columns and ian_col in df_copy.columns:
        df_copy["RATIO_IDA_IAN"] = df_copy[ida_col] / (df_copy[ian_col] + 1e-5)

    ieg_col = f"IEG_{base_year}"
    ips_col = f"IPS_{base_year}"
    if ieg_col in df_copy.columns and ips_col in df_copy.columns:
        df_copy["RATIO_IEG_IPS"] = df_copy[ieg_col] / (df_copy[ips_col] + 1e-5)

    inde_col = f"INDE_{base_year}"
    media_col = f"MEDIA_INDICADORES_{base_year}"
    if inde_col in df_copy.columns and media_col in df_copy.columns:
        df_copy["DIFF_INDE_MEDIA"] = df_copy[inde_col] - df_copy[media_col]

    return df_copy


def create_categorical_features(df: pd.DataFrame, base_year: int = BASE_YEAR) -> pd.DataFrame:
    df_copy = df.copy()

    def normalize_text(value: object) -> str:
        if pd.isna(value):
            return "desconhecido"
        text = str(value).strip().lower()
        text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8")
        return text

    pedra_col = f"PEDRA_{base_year}"
    if pedra_col in df_copy.columns:
        df_copy[f"PEDRA_{base_year}_NORM"] = df_copy[pedra_col].apply(normalize_text)

    pv_col = f"PONTO_VIRADA_{base_year}"
    if pv_col in df_copy.columns:
        df_copy[f"PONTO_VIRADA_{base_year}_NORM"] = df_copy[pv_col].apply(normalize_text)

    inst_col = f"INSTITUICAO_ENSINO_ALUNO_{base_year}"
    if inst_col in df_copy.columns:
        df_copy[f"INSTITUICAO_ENSINO_ALUNO_{base_year}_NORM"] = df_copy[inst_col].apply(
            normalize_text
        )

    fase_col = f"FASE_TURMA_{base_year}"
    if fase_col in df_copy.columns:
        df_copy[f"FASE_TURMA_{base_year}_NORM"] = df_copy[fase_col].apply(normalize_text)

    return df_copy


def create_age_features(df: pd.DataFrame, base_year: int = BASE_YEAR) -> pd.DataFrame:
    df_copy = df.copy()

    idade_col = f"IDADE_ALUNO_{base_year}"
    anos_pm_col = f"ANOS_PM_{base_year}"
    if idade_col in df_copy.columns:
        df_copy["FAIXA_ETARIA"] = pd.cut(
            df_copy[idade_col],
            bins=[0, 10, 13, 16, 100],
            labels=[0, 1, 2, 3],
        ).astype(float)

    if anos_pm_col in df_copy.columns and idade_col in df_copy.columns:
        df_copy["RATIO_ANOS_PM_IDADE"] = df_copy[anos_pm_col] / (df_copy[idade_col] + 1e-5)

    return df_copy


def select_features_for_training(df: pd.DataFrame, base_year: int = BASE_YEAR) -> pd.DataFrame:
    feature_cols = [
        f"INDE_{base_year}",
        f"IAA_{base_year}",
        f"IEG_{base_year}",
        f"IPS_{base_year}",
        f"IDA_{base_year}",
        f"IPP_{base_year}",
        f"IPV_{base_year}",
        f"IAN_{base_year}",
        f"MEDIA_INDICADORES_{base_year}",
        f"STD_INDICADORES_{base_year}",
        f"MIN_INDICADORES_{base_year}",
        f"MAX_INDICADORES_{base_year}",
        "RATIO_IDA_IAN",
        "RATIO_IEG_IPS",
        "DIFF_INDE_MEDIA",
        f"PEDRA_{base_year}_NORM",
        f"PONTO_VIRADA_{base_year}_NORM",
        f"INSTITUICAO_ENSINO_ALUNO_{base_year}_NORM",
        f"FASE_TURMA_{base_year}_NORM",
        f"IDADE_ALUNO_{base_year}",
        f"ANOS_PM_{base_year}",
        "FAIXA_ETARIA",
        "RATIO_ANOS_PM_IDADE",
        "RISCO_DEFASAGEM",
    ]

    available_cols = [col for col in feature_cols if col in df.columns]
    return df[available_cols].copy()


def engineer_features(df: pd.DataFrame, base_year: int = BASE_YEAR) -> pd.DataFrame:
    df_features = df.copy()

    df_features = create_basic_features(df_features, base_year=base_year)
    df_features = create_ratio_features(df_features, base_year=base_year)
    df_features = create_categorical_features(df_features, base_year=base_year)
    df_features = create_age_features(df_features, base_year=base_year)
    df_features = select_features_for_training(df_features, base_year=base_year)

    return df_features
