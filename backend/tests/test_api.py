import pytest
from fastapi.testclient import TestClient

from backend.src.api.main import app
from backend.tests.conftest import SAMPLE_INPUT_DICT


@pytest.fixture
def client(tmp_path):
    with TestClient(app) as c:
        from backend.src.monitoring.store import PredictionStore

        app.state.store = PredictionStore(tmp_path / "test.db")
        yield c


def test_health_check(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model_name" in data


def test_predict_returns_result(client: TestClient):
    response = client.post("/predict", json=SAMPLE_INPUT_DICT)
    assert response.status_code == 200
    data = response.json()
    assert "risco_defasagem" in data
    assert "probabilidade" in data
    assert "threshold" in data
    assert "classificacao" in data
    assert data["risco_defasagem"] in (0, 1)
    assert 0.0 <= data["probabilidade"] <= 1.0


def test_predict_input_invalido(client: TestClient):
    response = client.post("/predict", json={"INDE_2020": 6.5})
    assert response.status_code == 422


def test_prediction_stored_after_predict(client: TestClient):
    client.post("/predict", json=SAMPLE_INPUT_DICT)
    response = client.get("/predictions?limit=1")
    assert response.status_code == 200
    data = response.json()
    assert len(data) >= 1


def test_get_predictions_with_risco_filter(client: TestClient):
    client.post("/predict", json=SAMPLE_INPUT_DICT)
    response = client.get("/predictions?limit=10&risco=0")
    assert response.status_code == 200


def test_drift_insufficient_data(client: TestClient):
    response = client.get("/drift?n=50")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "insufficient_data"


def test_cluster_returns_result(client: TestClient):
    response = client.post("/cluster", json=SAMPLE_INPUT_DICT)
    assert response.status_code == 200
    data = response.json()
    assert "cluster_id" in data
    assert "perfil" in data
    assert "distancias" in data
    assert "descricao" in data
    assert data["perfil"] in ("risco_alto", "risco_moderado", "em_desenvolvimento", "alto_desempenho")
    assert isinstance(data["cluster_id"], int)
    assert len(data["distancias"]) == 4


def test_cluster_input_invalido(client: TestClient):
    response = client.post("/cluster", json={"INDE_2020": 6.5})
    assert response.status_code == 422
