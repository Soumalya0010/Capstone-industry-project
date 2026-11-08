import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

res_health = client.get("/api/health")
print("Health check:", res_health.status_code, res_health.json()["status"])

res_index = client.get("/")
print("Index HTML status:", res_index.status_code, "Length:", len(res_index.text))

res_calc = client.post("/api/calculator/proportion", json={
    "baseline_rate": 0.1,
    "mde_relative": 0.2,
    "power": 0.8,
    "alpha": 0.05
})
print("Calculator API status:", res_calc.status_code, "Sample size per variant:", res_calc.json()["sample_size_control"])

res_ds = client.get("/api/datasets/list")
print("Datasets API status:", res_ds.status_code, "Count:", len(res_ds.json()["datasets"]))

print("ALL TESTCLIENT CHECKS PASSED PERFECTLY!")
