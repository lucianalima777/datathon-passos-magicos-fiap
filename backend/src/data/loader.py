import pandas as pd
from pathlib import Path
from typing import Optional


def load_raw_data(file_path: Optional[Path] = None) -> pd.DataFrame:
    if file_path is None:
        from backend.src.config import RAW_DATA_PATH
        file_path = RAW_DATA_PATH
    
    df = pd.read_csv(file_path, sep=';', encoding='utf-8')
    return df

def load_processed_data(file_name: str = "processed_data.csv") -> pd.DataFrame:
    from backend.src.config import PROCESSED_DATA_DIR
    
    file_path = PROCESSED_DATA_DIR / file_name
    
    if not file_path.exists():
        raise FileNotFoundError(f"Arquivo {file_path} não encontrado")
    
    df = pd.read_csv(file_path)
    return df


def save_processed_data(df: pd.DataFrame, file_name: str = "processed_data.csv") -> None:
    from backend.src.config import PROCESSED_DATA_DIR
    
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    file_path = PROCESSED_DATA_DIR / file_name
    df.to_csv(file_path, index=False)
