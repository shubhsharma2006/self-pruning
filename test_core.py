"""
Test suite — Self-Pruning NN
Run: pytest test_core.py -v
"""
import math
import pytest
import torch
import numpy as np
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from train import PrunableLinear, SelfPruningNet, build_optimizer

# 1.  PrunableLinear — gradient flow

def test_prunable_linear_gradients():
    """Both weight and gate_scores must receive gradients."""
    layer = PrunableLinear(10, 5)
    x     = torch.randn(2, 10)
    loss  = layer(x).sum()
    loss.backward()
    assert layer.weight.grad      is not None, "weight.grad is None"
    assert layer.gate_scores.grad is not None, "gate_scores.grad is None"
    assert layer.bias.grad        is not None, "bias.grad is None"

def test_prunable_linear_output_shape():
    layer = PrunableLinear(16, 8)
    x     = torch.randn(4, 16)
    out   = layer(x)
    assert out.shape == (4, 8)

def test_gate_values_range():
    """Gate values must be in (0, 1) — they are sigmoid outputs."""
    layer = PrunableLinear(32, 16)
    gates = layer.gate_values()
    assert gates.min().item() >  0.0
    assert gates.max().item() <  1.0

def test_gate_init_near_half():
    """gate_scores initialised at 0 → sigmoid(0) = 0.5."""
    layer = PrunableLinear(64, 32)
    gates = layer.gate_values()
    assert abs(gates.mean().item() - 0.5) < 0.01

# 2.  SelfPruningNet

def test_sparsity_calculation():
    net   = SelfPruningNet(input_dim=64)
    stats = net.compute_sparsity()
    assert "overall"  in stats
    assert "layer_0"  in stats
    assert stats["overall"]["sparsity_%"] >= 0

def test_inference_shape():
    net = SelfPruningNet(input_dim=64)
    net.eval()
    with torch.no_grad():
        out = net(torch.randn(1, 64))
    assert out.shape == (1, 10)

def test_sparsity_loss_is_positive():
    net = SelfPruningNet(input_dim=64)
    sp  = net.sparsity_loss()
    assert sp.item() > 0

def test_all_gate_values_shape():
    net   = SelfPruningNet(input_dim=64)
    gates = net.all_gate_values()
    assert isinstance(gates, np.ndarray)
    assert gates.ndim == 1
    assert gates.shape[0] > 0

# 3.  Training loop — sparsity increases with higher lambda

def _quick_train(lam: float, steps: int = 30) -> float:
    """Train for a few steps on random data, return sparsity."""
    torch.manual_seed(42)
    model = SelfPruningNet(input_dim=32)
    opt   = build_optimizer(model, base_lr=1e-2, gate_lr_multiplier=10.0)
    crit  = torch.nn.CrossEntropyLoss()
    X     = torch.randn(64, 32)
    y     = torch.randint(0, 10, (64,))
    for _ in range(steps):
        opt.zero_grad()
        loss = crit(model(X), y) + lam * model.sparsity_loss()
        loss.backward()
        opt.step()
    return model.compute_sparsity()["overall"]["sparsity_%"]

def test_higher_lambda_more_sparse():
    """High lambda must produce equal or greater sparsity than low lambda."""
    sp_low  = _quick_train(lam=1e-5, steps=40)
    sp_high = _quick_train(lam=1e-1, steps=40)
    assert sp_high >= sp_low, f"Expected sp_high ({sp_high:.1f}%) >= sp_low ({sp_low:.1f}%)"

# 4.  FastAPI endpoints

@pytest.fixture
def client():
    from main import app, _registry, DEVICE
    from train import SelfPruningNet

    # Inject a mock model so tests don't need a checkpoint file
    mock_model = SelfPruningNet(input_dim=64).eval()
    _registry["test_model"] = {
        "model"    : mock_model,
        "lambda"   : 0.01,
        "accuracy" : 90.0,
        "sparsity" : 95.0,
        "input_dim": 64,
        "path"     : "fake/path.pt",
    }
    return TestClient(app)

API_KEY = "dev-secret-change-me"
HEADERS = {"X-API-Key": API_KEY}

def test_health_no_auth(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_list_models(client):
    r = client.get("/models", headers=HEADERS)
    assert r.status_code == 200
    assert "test_model" in r.json()

def test_predict_valid(client):
    r = client.post("/predict", headers=HEADERS,
                    json={"data": [0.1] * 64, "model_id": "test_model"})
    assert r.status_code == 200
    body = r.json()
    assert "prediction" in body
    assert "confidence" in body
    assert 0 <= body["confidence"] <= 1.0

def test_predict_wrong_dim(client):
    r = client.post("/predict", headers=HEADERS,
                    json={"data": [0.1] * 32, "model_id": "test_model"})
    assert r.status_code == 400

def test_predict_null_rejected(client):
    """None/null in data list must be rejected (not a valid float)."""
    # JSON null is the closest we can send to a non-numeric value
    data = [0.1] * 63 + [None]
    r = client.post("/predict", headers=HEADERS,
                    json={"data": data, "model_id": "test_model"})
    assert r.status_code == 422  # Pydantic validation error for wrong type

def test_predict_no_auth(client):
    r = client.post("/predict", json={"data": [0.1] * 64})
    assert r.status_code == 401

def test_stats_valid(client):
    r = client.get("/stats/test_model", headers=HEADERS)
    assert r.status_code == 200
    body = r.json()
    assert "layer_wise" in body
    assert "sparsity" in body

def test_stats_not_found(client):
    r = client.get("/stats/nonexistent", headers=HEADERS)
    assert r.status_code == 404
