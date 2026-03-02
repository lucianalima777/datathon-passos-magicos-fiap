from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

NUMERIC_DRIFT_COLS = [
    "INDE_2020",
    "IAA_2020",
    "IEG_2020",
    "IPS_2020",
    "IDA_2020",
    "IPP_2020",
    "IPV_2020",
    "IAN_2020",
]

DRIFT_THRESHOLD = 0.05

# Limiares PSI: <0.1 estavel, 0.1-0.2 alerta, >=0.2 critico
PSI_THRESHOLD_ALERT = 0.1
PSI_THRESHOLD_CRITICAL = 0.2


def _compute_psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """
    Calcula o Population Stability Index (PSI) entre distribuicoes de referencia e atual.

    PSI < 0.1  -> estavel (sem mudanca significativa)
    PSI < 0.2  -> alerta  (mudanca moderada)
    PSI >= 0.2 -> critico (mudanca significativa, investigar)
    """
    min_val = min(reference.min(), current.min())
    max_val = max(reference.max(), current.max())
    breakpoints = np.linspace(min_val, max_val, bins + 1)

    ref_counts, _ = np.histogram(reference, bins=breakpoints)
    cur_counts, _ = np.histogram(current, bins=breakpoints)

    epsilon = 1e-4
    ref_pct = ref_counts / len(reference)
    cur_pct = cur_counts / len(current)
    ref_pct = np.where(ref_pct == 0, epsilon, ref_pct)
    cur_pct = np.where(cur_pct == 0, epsilon, cur_pct)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def _psi_status(psi: float) -> str:
    if psi < PSI_THRESHOLD_ALERT:
        return "estavel"
    if psi < PSI_THRESHOLD_CRITICAL:
        return "alerta"
    return "critico"


def check_data_drift(recent_inputs: list[dict], reference: pd.DataFrame) -> dict:
    """
    Compara distribuicao de inputs recentes vs dados de treino usando KS test e PSI.

    A referencia deve ser carregada uma unica vez no startup da aplicacao e
    injetada aqui — evitando leitura de disco a cada requisicao.

    Args:
        recent_inputs: Lista de dicts com os inputs das predicoes recentes.
        reference: DataFrame de referencia (dados de treino) com NUMERIC_DRIFT_COLS.

    Returns:
        Dict com status, drift_share, n_drifted_columns e detalhes por coluna.
    """
    available_cols = [c for c in NUMERIC_DRIFT_COLS if c in reference.columns]
    current = pd.DataFrame(recent_inputs)
    available_cols = [c for c in available_cols if c in current.columns]

    column_details = {}
    n_drifted = 0

    for col in available_cols:
        ref_values = reference[col].dropna().to_numpy()
        cur_values = current[col].dropna().to_numpy()
        ks_stat, p_value = stats.ks_2samp(ref_values, cur_values)
        psi = _compute_psi(ref_values, cur_values)
        drifted = bool(p_value < DRIFT_THRESHOLD)
        if drifted:
            n_drifted += 1
        column_details[col] = {
            "ks_statistic": round(float(ks_stat), 4),
            "ks_p_value": round(float(p_value), 4),
            "psi": round(psi, 4),
            "psi_status": _psi_status(psi),
            "drifted": drifted,
        }

    n_cols = len(available_cols)
    drift_share = n_drifted / n_cols if n_cols > 0 else 0.0

    return {
        "status": "completed",
        "dataset_drift": drift_share > 0.5,
        "drift_share": round(drift_share, 4),
        "n_drifted_columns": n_drifted,
        "n_features": n_cols,
        "n_reference": len(reference),
        "n_current": len(current),
        "column_details": column_details,
    }
