from backend.src.monitoring.store import PredictionStore


def test_save_and_retrieve(prediction_store: PredictionStore, sample_input: dict):
    row_id = prediction_store.save_prediction(
        input_data=sample_input,
        prediction=1,
        probability=0.85,
        threshold=0.3,
        model_name="test_model",
        response_time_ms=12.5,
    )
    assert row_id == 1

    rows = prediction_store.get_predictions(limit=10)
    assert len(rows) == 1
    assert rows[0]["prediction"] == 1
    assert rows[0]["probability"] == 0.85


def test_filter_by_prediction(prediction_store: PredictionStore, sample_input: dict):
    prediction_store.save_prediction(
        input_data=sample_input, prediction=0, probability=0.2,
        threshold=0.3, model_name="m",
    )
    prediction_store.save_prediction(
        input_data=sample_input, prediction=1, probability=0.9,
        threshold=0.3, model_name="m",
    )

    risco = prediction_store.get_predictions(limit=10, prediction=1)
    assert len(risco) == 1
    assert risco[0]["prediction"] == 1

    adequado = prediction_store.get_predictions(limit=10, prediction=0)
    assert len(adequado) == 1


def test_get_recent_inputs(prediction_store: PredictionStore, sample_input: dict):
    for _ in range(5):
        prediction_store.save_prediction(
            input_data=sample_input, prediction=0, probability=0.2,
            threshold=0.3, model_name="m",
        )

    inputs = prediction_store.get_recent_inputs(n=3)
    assert len(inputs) == 3
    assert "INDE_2020" in inputs[0]


def test_count(prediction_store: PredictionStore, sample_input: dict):
    assert prediction_store.count() == 0
    prediction_store.save_prediction(
        input_data=sample_input, prediction=0, probability=0.2,
        threshold=0.3, model_name="m",
    )
    assert prediction_store.count() == 1
