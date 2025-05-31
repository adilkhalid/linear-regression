from fastapi.testclient import TestClient
from simple_linear_regression.main import app  # assuming app is defined in main.py

client = TestClient(app)


def test_predict_endpoint():
    response = client.post("/predict/?medinc=5.0")
    assert response.status_code == 200
    data = response.json()
    assert "predicted_house_value_100k" in data
    assert isinstance(data["predicted_house_value_100k"], float)

