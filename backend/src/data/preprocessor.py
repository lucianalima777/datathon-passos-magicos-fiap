import pandas as pd
from typing import List, Optional


def convert_numeric_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    df_copy = df.copy()
    
    for col in columns:
        if col in df_copy.columns:
            df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
    
    return df_copy


def create_target(df: pd.DataFrame, defasagem_col: str) -> pd.DataFrame:
    df_copy = df.copy()
    if defasagem_col not in df_copy.columns:
        raise ValueError(f"Coluna {defasagem_col} não encontrada")

    df_copy["RISCO_DEFASAGEM"] = (df_copy[defasagem_col] < 0).astype(int)
    return df_copy


def remove_rows_without_defasagem(
    df: pd.DataFrame, defasagem_col: str
) -> pd.DataFrame:
    if defasagem_col not in df.columns:
        raise ValueError(f"Coluna {defasagem_col} não encontrada")

    df_filtered = df[df[defasagem_col].notna()].copy()
    return df_filtered


def handle_missing_values(
    df: pd.DataFrame, strategy: str = "median", indicator_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    from backend.src.config import BASE_YEAR, indicadores_year
    
    df_copy = df.copy()
    if indicator_cols is None:
        indicator_cols = indicadores_year(BASE_YEAR)

    numeric_cols = [col for col in indicator_cols if col in df_copy.columns]
    
    for col in numeric_cols:
        if df_copy[col].isna().any():
            if strategy == 'median':
                fill_value = df_copy[col].median()
            elif strategy == 'mean':
                fill_value = df_copy[col].mean()
            else:
                fill_value = 0
            
            df_copy[col] = df_copy[col].fillna(fill_value)
    
    return df_copy


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    from backend.src.config import COLUNAS_NUMERICAS, DEFASAGEM_COL

    df_processed = df.copy()

    df_processed = convert_numeric_columns(df_processed, COLUNAS_NUMERICAS)
    df_processed = remove_rows_without_defasagem(df_processed, DEFASAGEM_COL)
    df_processed = create_target(df_processed, DEFASAGEM_COL)

    return df_processed
