import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.src.data.loader import load_raw_data, save_processed_data
from backend.src.data.preprocessor import preprocess_data
from backend.src.features.engineer import engineer_features


def main():
    print("Carregando dados brutos...")
    df_raw = load_raw_data()
    print(f"Dados carregados: {df_raw.shape}")
    
    print("\nPré-processando dados...")
    df_preprocessed = preprocess_data(df_raw)
    print(f"Dados após pré-processamento: {df_preprocessed.shape}")
    
    print("\nCriando features...")
    df_final = engineer_features(df_preprocessed)
    print(f"Dados finais: {df_final.shape}")
    
    print(f"\nColunas finais: {list(df_final.columns)}")
    print(f"\nDistribuição do target:")
    print(df_final['RISCO_DEFASAGEM'].value_counts())
    print(f"Proporção em risco: {df_final['RISCO_DEFASAGEM'].mean()*100:.1f}%")
    
    print("\nSalvando dados processados...")
    save_processed_data(df_final, "train_data.csv")
    print("Dados salvos com sucesso!")
    
    print("\nInfo do dataset final:")
    print(df_final.info())


if __name__ == "__main__":
    main()
